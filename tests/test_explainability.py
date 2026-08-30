from src.explainability.explain_predictions import explain_complaint


def main() -> None:
    complaint_text = (
        "Hostel Block B la water leak aagudhu and floor slippery ah iruku."
    )

    priority_features = {
        "affected_population": 160,
        "safety_flag": 1,
        "repeat_count": 2,
        "category": "Water and Plumbing",
        "location_type": "Hostel",
        "language": "ta_en",
    }

    result = explain_complaint(
        complaint_text=complaint_text,
        priority_features=priority_features,
    )

    print("\nCategory explanation:")
    print(result["category_explanation"])

    print("\nPriority explanation:")
    print(result["priority_explanation"])

    print("\nHuman-readable explanation:")
    print(result["narrative"])


if __name__ == "__main__":
    main()