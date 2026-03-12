import spikeinterface.full as si
import numpy as np
import kyutils
from pathlib import Path
from typing import Tuple
import ripple_detection as rd
import scipy
import position_tools as pt


import argparse

animal_name = "L14"
data_path = Path("/nimbus/kyu") / animal_name
date = "20240611"
analysis_path = data_path / "singleday_sort" / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)


def consolidate_sorting():
    sorting_list_v1 = []
    for probe_idx in [0, 3]:
        print(f"probe{probe_idx}")
        for shank_idx in range(4):
            print(f"shank{shank_idx}")
            sorting_curations_path = Path(
                f"/home/kyu/repos/sorting-curations/khl02007/L14/{date}/probe{probe_idx}/shank{shank_idx}/ms4/curation.json"
            )

            if not sorting_curations_path.exists():
                continue

            sorting_path = (
                analysis_path
                / "sorting"
                / f"sorting_probe{probe_idx}_shank{shank_idx}_ms4"
            )

            sorting = si.NpzSortingExtractor(
                sorting_path / "sorter_output" / "firings.npz"
            )
            print(
                f"{sorting.get_num_units()} units in probe {probe_idx} shank {shank_idx}"
            )

            curated_sorting = si.apply_sortingview_curation(
                sorting,
                sorting_curations_path,
                exclude_labels=["noise", "reject"],
                include_labels=None,
                skip_merge=False,
            )
            curated_sorting = si.remove_duplicated_spikes(
                curated_sorting, censored_period_ms=0.1
            )
            curated_sorting = si.remove_redundant_units(
                curated_sorting,
                duplicate_threshold=0.9,
                remove_strategy="max_spikes",
                align=False,
            )

            print(
                f"after curation {curated_sorting.get_num_units()} units in probe {probe_idx} shank {shank_idx}"
            )

            curated_sorting.save_to_folder(
                analysis_path
                / "sorting"
                / f"curated_sorting_probe{probe_idx}_shank{shank_idx}_ms4",
                overwrite=True,
            )

            sorting_list_v1.append(curated_sorting)

    sorting_v1 = si.UnitsAggregationSorting(sorting_list_v1)
    sorting_v1.save_to_folder(analysis_path / "sorting_v1", overwrite=True)
    print("done with v1")

    sorting_list_ca1 = []
    for probe_idx in [1, 2]:
        print(f"probe{probe_idx}")
        for shank_idx in range(4):
            print(f"shank{shank_idx}")

            sorting_curations_path = Path(
                f"/home/kyu/repos/sorting-curations/khl02007/L14/{date}/probe{probe_idx}/shank{shank_idx}/ms4/curation.json"
            )

            if not sorting_curations_path.exists():
                continue

            sorting_path = (
                analysis_path
                / "sorting"
                / f"sorting_probe{probe_idx}_shank{shank_idx}_ms4"
            )
            sorting = si.NpzSortingExtractor(
                sorting_path / "sorter_output" / "firings.npz"
            )
            print(
                f"{sorting.get_num_units()} units in probe {probe_idx} shank {shank_idx}"
            )

            curated_sorting = si.apply_sortingview_curation(
                sorting,
                sorting_curations_path,
                exclude_labels=["noise", "reject"],
                include_labels=None,
                skip_merge=False,
            )
            curated_sorting = si.remove_duplicated_spikes(
                curated_sorting, censored_period_ms=0.1
            )
            curated_sorting = si.remove_redundant_units(
                curated_sorting,
                duplicate_threshold=0.9,
                remove_strategy="max_spikes",
                align=False,
            )

            print(
                f"after curation {curated_sorting.get_num_units()} units in probe {probe_idx} shank {shank_idx}"
            )

            curated_sorting.save_to_folder(
                analysis_path
                / "sorting"
                / f"curated_sorting_probe{probe_idx}_shank{shank_idx}_ms4",
                overwrite=True,
            )

            sorting_list_ca1.append(curated_sorting)

    sorting_ca1 = si.UnitsAggregationSorting(sorting_list_ca1)
    sorting_ca1.save_to_folder(analysis_path / "sorting_ca1", overwrite=True)
    print("done with ca1")

    return sorting_v1, sorting_ca1


def main():
    consolidate_sorting()


if __name__ == "__main__":
    main()
