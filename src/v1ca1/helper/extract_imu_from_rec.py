from __future__ import annotations

"""Extract true IMU update events from one epoch REC file and save a parquet table."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    get_analysis_path,
    load_ephys_timestamps_by_epoch,
)

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_OUTPUT_NAME_TEMPLATE = "{epoch}_imu.parquet"
ACCEL_CHANNELS = [
    "Headstage_AccelX",
    "Headstage_AccelY",
    "Headstage_AccelZ",
]
GYRO_CHANNELS = [
    "Headstage_GyroX",
    "Headstage_GyroY",
    "Headstage_GyroZ",
]
ACCEL_SCALE_G_PER_LSB = 0.000061
GYRO_SCALE_DEG_S_PER_LSB = 0.061
IMU_TABLE_COLUMNS = [
    "accel_time",
    "gyro_time",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "gyro_x_deg_s",
    "gyro_y_deg_s",
    "gyro_z_deg_s",
]


class RecImuAccessor:
    """Read multiplexed raw IMU channels from a Trodes REC file."""

    def __init__(
        self,
        root: ET.Element,
        raw_memmap: np.ndarray,
        channel_info: dict[str, dict[str, int]],
        num_packets: int,
        sampling_rate: float,
    ) -> None:
        self.root = root
        self.raw_memmap = raw_memmap
        self.channel_info = channel_info
        self.num_packets = num_packets
        self.sampling_rate = sampling_rate

    @classmethod
    def from_rec_path(cls, rec_path: Path) -> "RecImuAccessor":
        """Build an accessor for one REC file."""
        with open(rec_path, "rb") as file:
            while True:
                line = file.readline()
                if not line:
                    raise RuntimeError("Could not find rec XML header end marker.")
                if b"</Configuration>" in line:
                    header_size = file.tell()
                    break
            file.seek(0)
            root = ET.fromstring(file.read(header_size).decode("utf8"))

        hardware_config = root.find("HardwareConfiguration")
        spike_config = root.find("SpikeConfiguration")
        if hardware_config is None or spike_config is None:
            raise RuntimeError("rec header missing hardware or spike configuration.")

        num_chip_channels = int(hardware_config.attrib["numChannels"])
        spike_config_channels = sum(len(group) for group in spike_config)
        num_ephys_channels = min(num_chip_channels, spike_config_channels)

        packet_size = 1
        device_bytes: dict[str, int] = {}
        for device in hardware_config:
            device_bytes[device.attrib["name"]] = packet_size
            packet_size += int(device.attrib["numBytes"])
        packet_size += 4 + (2 * num_ephys_channels)

        raw_memmap = np.memmap(
            rec_path,
            mode="r",
            offset=header_size,
            dtype="<u1",
        )
        num_packets = raw_memmap.size // packet_size
        raw_memmap = raw_memmap[: num_packets * packet_size].reshape(-1, packet_size)

        if "Multiplexed" in device_bytes:
            multiplexed_start = device_bytes["Multiplexed"]
        elif "headstageSensor" in device_bytes:
            multiplexed_start = device_bytes["headstageSensor"]
        else:
            raise RuntimeError("Could not find multiplexed/headstage sensor bytes.")

        channel_info: dict[str, dict[str, int]] = {}
        for device in hardware_config:
            if device.attrib.get("name") not in {"Multiplexed", "headstageSensor"}:
                continue
            for channel in device:
                if channel.attrib.get("dataType") != "analog":
                    continue
                channel_info[channel.attrib["id"]] = {
                    "data_offset": multiplexed_start + int(channel.attrib["startByte"]),
                    "idbyte_offset": multiplexed_start
                    + int(channel.attrib["interleavedDataIDByte"]),
                    "idbit": int(channel.attrib["interleavedDataIDBit"]),
                }

        return cls(
            root=root,
            raw_memmap=raw_memmap,
            channel_info=channel_info,
            num_packets=num_packets,
            sampling_rate=float(hardware_config.attrib["samplingRate"]),
        )

    def _get_channel_group_info(self, channel_names: list[str]) -> tuple[int, int]:
        """Return the shared update-byte offset and bit for one sensor group."""
        missing_channels = [name for name in channel_names if name not in self.channel_info]
        if missing_channels:
            raise ValueError(
                "REC file is missing required IMU channels: "
                f"{missing_channels!r}."
            )

        idbyte_offsets = {self.channel_info[name]["idbyte_offset"] for name in channel_names}
        idbits = {self.channel_info[name]["idbit"] for name in channel_names}
        if len(idbyte_offsets) != 1 or len(idbits) != 1:
            raise ValueError(
                "Expected one shared update selector for the IMU channel group, found "
                f"idbyte_offsets={sorted(idbyte_offsets)!r}, idbits={sorted(idbits)!r}."
            )
        return int(next(iter(idbyte_offsets))), int(next(iter(idbits)))

    def get_update_indices(self, channel_names: list[str]) -> np.ndarray:
        """Return packet indices where one sensor group emitted a new sample."""
        idbyte_offset, idbit = self._get_channel_group_info(channel_names)
        updates = ((self.raw_memmap[:, idbyte_offset] >> idbit) & 1).astype(bool)
        update_indices = np.flatnonzero(updates)
        if update_indices.size == 0:
            raise ValueError(
                "REC file does not contain any update events for IMU channels "
                f"{channel_names!r}."
            )
        return update_indices

    def read_channel_values_at_indices(
        self,
        indices: np.ndarray,
        channel_names: list[str],
    ) -> np.ndarray:
        """Decode raw channel values at specific packet indices."""
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional.")
        if indices.size == 0:
            return np.empty((0, len(channel_names)), dtype=np.int16)

        rows = self.raw_memmap[indices]
        values = np.empty((indices.size, len(channel_names)), dtype=np.int16)
        for column_index, channel_name in enumerate(channel_names):
            info = self.channel_info[channel_name]
            values[:, column_index] = (
                rows[:, info["data_offset"]].astype(np.int16)
                + rows[:, info["data_offset"] + 1].astype(np.int16) * 256
            ).astype(np.int16)
        return values

    def extract_sensor_group(self, channel_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return update packet indices and raw values for one IMU sensor group."""
        update_indices = self.get_update_indices(channel_names)
        values = self.read_channel_values_at_indices(update_indices, channel_names)
        return update_indices, values


def get_default_output_name(epoch: str) -> str:
    """Return the default parquet filename for one epoch."""
    return DEFAULT_OUTPUT_NAME_TEMPLATE.format(epoch=epoch)


def load_epoch_timestamps(analysis_path: Path, epoch: str) -> tuple[np.ndarray, str]:
    """Load the saved ephys timestamps for one epoch."""
    _epoch_tags, timestamps_by_epoch, source = load_ephys_timestamps_by_epoch(analysis_path)
    if epoch not in timestamps_by_epoch:
        available_epochs = sorted(timestamps_by_epoch)
        raise ValueError(
            f"Epoch {epoch!r} not found in saved ephys timestamps. "
            f"Available epochs: {available_epochs!r}"
        )
    return np.asarray(timestamps_by_epoch[epoch], dtype=float), source


def extract_true_imu_events(rec_path: Path) -> dict[str, np.ndarray | float | int]:
    """Extract true accel and gyro update events from one REC file."""
    accessor = RecImuAccessor.from_rec_path(rec_path)
    if accessor.num_packets <= 0:
        raise ValueError(f"REC file does not contain any packets: {rec_path}")

    accel_packet_indices, accel_samples = accessor.extract_sensor_group(ACCEL_CHANNELS)
    gyro_packet_indices, gyro_samples = accessor.extract_sensor_group(GYRO_CHANNELS)
    return {
        "num_packets": int(accessor.num_packets),
        "packet_sampling_rate_hz": float(accessor.sampling_rate),
        "accel_packet_indices": accel_packet_indices,
        "accel_samples": accel_samples,
        "gyro_packet_indices": gyro_packet_indices,
        "gyro_samples": gyro_samples,
    }


def get_group_timestamps(
    packet_timestamps: np.ndarray,
    packet_indices: np.ndarray,
    label: str,
) -> np.ndarray:
    """Map packet indices for one sensor group onto saved ephys timestamps."""
    packet_timestamps = np.asarray(packet_timestamps, dtype=float)
    packet_indices = np.asarray(packet_indices, dtype=np.int64)

    if packet_timestamps.ndim != 1:
        raise ValueError("Epoch timestamps must be one-dimensional.")
    if packet_indices.ndim != 1:
        raise ValueError(f"{label} packet indices must be one-dimensional.")
    if packet_indices.size == 0:
        raise ValueError(f"{label} packet indices are empty.")
    if packet_indices[0] < 0 or packet_indices[-1] >= packet_timestamps.size:
        raise ValueError(
            f"{label} packet indices fall outside the epoch timestamp vector: "
            f"max index {int(packet_indices[-1])}, timestamp count {packet_timestamps.size}."
        )
    return packet_timestamps[packet_indices]


def validate_paired_imu_packets(
    accel_packet_indices: np.ndarray,
    gyro_packet_indices: np.ndarray,
) -> np.ndarray:
    """Validate that accel and gyro update events can be paired by order."""
    accel_packet_indices = np.asarray(accel_packet_indices, dtype=np.int64)
    gyro_packet_indices = np.asarray(gyro_packet_indices, dtype=np.int64)

    if accel_packet_indices.ndim != 1 or gyro_packet_indices.ndim != 1:
        raise ValueError("Accel and gyro packet indices must both be one-dimensional.")
    if accel_packet_indices.size != gyro_packet_indices.size:
        raise ValueError(
            "Accel and gyro update counts do not match: "
            f"{accel_packet_indices.size} vs {gyro_packet_indices.size}."
        )
    if accel_packet_indices.size == 0:
        raise ValueError("Accel and gyro update indices are empty.")

    packet_offsets = gyro_packet_indices - accel_packet_indices
    if accel_packet_indices.size == 1:
        return packet_offsets

    accel_step = float(np.median(np.diff(accel_packet_indices)))
    gyro_step = float(np.median(np.diff(gyro_packet_indices)))
    max_expected_offset = max(1.0, 0.5 * min(accel_step, gyro_step))
    if np.max(np.abs(packet_offsets)) > max_expected_offset:
        raise ValueError(
            "Accel and gyro update streams are too far apart to pair by order. "
            f"Max packet offset {int(np.max(np.abs(packet_offsets)))} exceeds "
            f"the expected bound {max_expected_offset:.1f}."
        )
    return packet_offsets


def scale_accel_samples(accel_samples: np.ndarray) -> np.ndarray:
    """Convert raw accelerometer LSB counts to g."""
    return np.asarray(accel_samples, dtype=float) * ACCEL_SCALE_G_PER_LSB


def scale_gyro_samples(gyro_samples: np.ndarray) -> np.ndarray:
    """Convert raw gyroscope LSB counts to deg/s."""
    return np.asarray(gyro_samples, dtype=float) * GYRO_SCALE_DEG_S_PER_LSB


def build_imu_dataframe(
    accel_timestamps: np.ndarray,
    accel_samples: np.ndarray,
    gyro_timestamps: np.ndarray,
    gyro_samples: np.ndarray,
) -> "pd.DataFrame":
    """Build a parquet-ready IMU table from paired true update events."""
    import pandas as pd

    accel_timestamps = np.asarray(accel_timestamps, dtype=float)
    accel_samples = np.asarray(accel_samples)
    gyro_timestamps = np.asarray(gyro_timestamps, dtype=float)
    gyro_samples = np.asarray(gyro_samples)

    if accel_timestamps.ndim != 1 or gyro_timestamps.ndim != 1:
        raise ValueError("Accel and gyro timestamps must both be one-dimensional.")
    if accel_samples.ndim != 2 or gyro_samples.ndim != 2:
        raise ValueError("Accel and gyro samples must both be two-dimensional arrays.")
    if accel_samples.shape[1] != len(ACCEL_CHANNELS):
        raise ValueError(
            "Accel samples must have three columns. "
            f"Found shape {accel_samples.shape}."
        )
    if gyro_samples.shape[1] != len(GYRO_CHANNELS):
        raise ValueError(
            "Gyro samples must have three columns. "
            f"Found shape {gyro_samples.shape}."
        )
    if accel_timestamps.size != accel_samples.shape[0]:
        raise ValueError(
            "Accel timestamp count does not match the accel sample count: "
            f"{accel_timestamps.size} vs {accel_samples.shape[0]}."
        )
    if gyro_timestamps.size != gyro_samples.shape[0]:
        raise ValueError(
            "Gyro timestamp count does not match the gyro sample count: "
            f"{gyro_timestamps.size} vs {gyro_samples.shape[0]}."
        )
    if accel_timestamps.size != gyro_timestamps.size:
        raise ValueError(
            "Accel and gyro true sample counts do not match: "
            f"{accel_timestamps.size} vs {gyro_timestamps.size}."
        )

    accel_samples_g = scale_accel_samples(accel_samples)
    gyro_samples_deg_s = scale_gyro_samples(gyro_samples)
    return pd.DataFrame(
        {
            "accel_time": accel_timestamps,
            "gyro_time": gyro_timestamps,
            "accel_x_g": accel_samples_g[:, 0],
            "accel_y_g": accel_samples_g[:, 1],
            "accel_z_g": accel_samples_g[:, 2],
            "gyro_x_deg_s": gyro_samples_deg_s[:, 0],
            "gyro_y_deg_s": gyro_samples_deg_s[:, 1],
            "gyro_z_deg_s": gyro_samples_deg_s[:, 2],
        },
        columns=IMU_TABLE_COLUMNS,
    )


def save_parquet_output(table: "pd.DataFrame", output_path: Path) -> Path:
    """Write one parquet table with a clear dependency error."""
    try:
        table.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Saving parquet outputs requires `pyarrow` or `fastparquet` to be installed."
        ) from exc
    return output_path


def extract_imu_from_rec(
    rec_path: Path,
    animal_name: str,
    date: str,
    epoch: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_name: str | None = None,
) -> Path:
    """Extract true IMU update events from one REC file and save them as parquet."""
    if not rec_path.exists():
        raise FileNotFoundError(f"REC file not found: {rec_path}")

    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    print(f"Processing {animal_name} {date} epoch {epoch}.")
    packet_timestamps, timestamp_source = load_epoch_timestamps(
        analysis_path=analysis_path,
        epoch=epoch,
    )
    imu_events = extract_true_imu_events(rec_path)

    num_packets = int(imu_events["num_packets"])
    if packet_timestamps.size != num_packets:
        raise ValueError(
            "Epoch ephys timestamp count does not match the REC packet count: "
            f"{packet_timestamps.size} vs {num_packets}."
        )

    accel_packet_indices = np.asarray(imu_events["accel_packet_indices"], dtype=np.int64)
    gyro_packet_indices = np.asarray(imu_events["gyro_packet_indices"], dtype=np.int64)
    accel_samples = np.asarray(imu_events["accel_samples"])
    gyro_samples = np.asarray(imu_events["gyro_samples"])
    packet_offsets = validate_paired_imu_packets(
        accel_packet_indices=accel_packet_indices,
        gyro_packet_indices=gyro_packet_indices,
    )

    accel_timestamps = get_group_timestamps(
        packet_timestamps=packet_timestamps,
        packet_indices=accel_packet_indices,
        label="Accel",
    )
    gyro_timestamps = get_group_timestamps(
        packet_timestamps=packet_timestamps,
        packet_indices=gyro_packet_indices,
        label="Gyro",
    )
    table = build_imu_dataframe(
        accel_timestamps=accel_timestamps,
        accel_samples=accel_samples,
        gyro_timestamps=gyro_timestamps,
        gyro_samples=gyro_samples,
    )

    output_path = analysis_path / (
        output_name if output_name is not None else get_default_output_name(epoch)
    )
    saved_path = save_parquet_output(table=table, output_path=output_path)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.helper.extract_imu_from_rec",
        parameters={
            "rec_path": rec_path,
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "data_root": data_root,
            "output_name": output_name,
        },
        outputs={
            "timestamp_source": timestamp_source,
            "packet_sampling_rate_hz": float(imu_events["packet_sampling_rate_hz"]),
            "packet_count": num_packets,
            "accel_true_sample_count": int(accel_timestamps.size),
            "gyro_true_sample_count": int(gyro_timestamps.size),
            "packet_offset_min": int(packet_offsets.min()),
            "packet_offset_max": int(packet_offsets.max()),
            "accel_unit": "g",
            "gyro_unit": "deg/s",
            "parquet_path": saved_path,
        },
    )
    print(f"Saved IMU parquet to {saved_path}")
    print(f"Saved run metadata to {log_path}")
    return saved_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the REC IMU extractor."""
    parser = argparse.ArgumentParser(
        description="Extract true raw IMU update events from one REC file"
    )
    parser.add_argument(
        "--rec-path",
        type=Path,
        required=True,
        help="Path to the single-epoch REC file.",
    )
    parser.add_argument(
        "--animal-name",
        required=True,
        help="Animal name",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Recording date in YYYYMMDD format",
    )
    parser.add_argument(
        "--epoch",
        required=True,
        help="Epoch label whose saved ephys timestamps should be used to timestamp true IMU updates.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional parquet filename written under the session analysis directory. "
            "Default: '{epoch}_imu.parquet'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the REC IMU extraction CLI."""
    args = parse_arguments()
    extract_imu_from_rec(
        rec_path=args.rec_path,
        animal_name=args.animal_name,
        date=args.date,
        epoch=args.epoch,
        data_root=args.data_root,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
