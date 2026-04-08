from src.evaluation.model_comparison import ModelComparison


def main():
    comparison = ModelComparison()

    df = comparison.get_comparison()

    print("\n🏆 AUTO MODEL COMPARISON\n")
    print(df)

    comparison.save_comparison_results()

    print("\n✅ Comparison results saved in artifacts/results/")


if __name__ == "__main__":
    main()