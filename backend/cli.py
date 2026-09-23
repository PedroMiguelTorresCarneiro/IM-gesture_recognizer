from __future__ import annotations

import argparse

import uvicorn

from backend.feature_extraction.pipeline import (
    extract_all_gestures,
    extract_gesture,
    print_results,
)


# ============================================================
# SERVE
# ============================================================

def command_serve(
    args: argparse.Namespace,
) -> None:
    """
    Inicia o servidor FastAPI.
    """

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


# ============================================================
# EXTRACT
# ============================================================

def command_extract(
    args: argparse.Namespace,
) -> None:
    """
    Executa Feature Extraction.
    """

    # --------------------------------------------------------
    # Single gesture
    # --------------------------------------------------------

    if args.gesture:

        print()
        print(
            f"Extracting gesture: "
            f"{args.gesture}"
        )
        print()

        results = extract_gesture(
            args.gesture,
            force=args.force,
        )

        print_results(
            results
        )

        return

    # --------------------------------------------------------
    # All gestures
    # --------------------------------------------------------

    if args.all:

        print()
        print(
            "Extracting all gestures"
        )
        print()

        all_results = (
            extract_all_gestures(
                force=args.force,
            )
        )

        for (
            gesture_name,
            results,
        ) in all_results.items():

            print()
            print(
                "================================"
            )
            print(
                f"Gesture: {gesture_name}"
            )
            print(
                "================================"
            )

            print_results(
                results
            )

        return


# ============================================================
# PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Cria o parser principal do Gesture Capture CLI.
    """

    parser = argparse.ArgumentParser(
        prog="gesture-capture",
        description=(
            "Gesture Capture dataset and "
            "feature extraction tools."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ========================================================
    # SERVE
    # ========================================================

    serve_parser = subparsers.add_parser(
        "serve",
        help=(
            "Start the Gesture Capture web application."
        ),
    )

    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Host address. "
            "Default: 127.0.0.1"
        ),
    )

    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help=(
            "HTTP port. "
            "Default: 8000"
        ),
    )

    serve_parser.add_argument(
        "--no-reload",
        action="store_true",
        help=(
            "Disable automatic development reload."
        ),
    )

    serve_parser.set_defaults(
        func=command_serve
    )

    # ========================================================
    # EXTRACT
    # ========================================================

    extract_parser = subparsers.add_parser(
        "extract",
        help=(
            "Extract MediaPipe gesture features."
        ),
    )

    extract_target = (
        extract_parser.add_mutually_exclusive_group(
            required=True
        )
    )

    extract_target.add_argument(
        "--gesture",
        type=str,
        metavar="NAME",
        help=(
            "Extract a single gesture, "
            "e.g. swipe-left."
        ),
    )

    extract_target.add_argument(
        "--all",
        action="store_true",
        help=(
            "Extract all gestures."
        ),
    )

    extract_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing feature extraction."
        ),
    )

    extract_parser.set_defaults(
        func=command_extract
    )

    return parser


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Gesture Capture CLI entry point.
    """

    parser = build_parser()

    args = parser.parse_args()

    args.func(
        args
    )


if __name__ == "__main__":
    main()