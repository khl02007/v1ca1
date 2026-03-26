"""Compatibility wrapper for the modern theta-phase CLI."""

from v1ca1.oscillation.get_theta_phase import (
    get_theta_phase_for_session,
    main,
    parse_arguments,
)

get_theta_phase = get_theta_phase_for_session

__all__ = ["get_theta_phase", "get_theta_phase_for_session", "main", "parse_arguments"]


if __name__ == "__main__":
    main()
