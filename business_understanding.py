from rich.console import Console

console = Console()


def main():
    console.rule("Business Understanding")
    console.print(
        "Goal: Classify images into three classes: Black, Brown, White.", style="bold"
    )
    console.print("Primary metric: Validation accuracy")
    console.print("Constraints: Small project, simple baseline model")
    console.print("Success criteria: >70% validation accuracy and usable predictor")


if __name__ == "__main__":
    main()