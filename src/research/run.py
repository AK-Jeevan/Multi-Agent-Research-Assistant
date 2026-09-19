import argparse

from src.research.workflow import run_research


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2 research workflow.")
    parser.add_argument("question")
    args = parser.parse_args()
    result = run_research(args.question)
    print(result["draft"])
    print("\nCritic review:")
    print(result["critique"])
    print(f"\nSaved {len(result['trajectory'])} trajectory events.")


if __name__ == "__main__":
    main()
