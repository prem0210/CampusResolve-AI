from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TOTAL_UNIQUE_COMPLAINTS = 3000
DUPLICATE_RATE = 0.30
OUTPUT_PATH = Path("data/processed/complaints_master_synthetic.csv")

random.seed(SEED)
rng = np.random.default_rng(SEED)

CATEGORY_CONFIG = {
    "Electrical and Power": {
        "department": "Electrical Maintenance",
        "location_types": ["Hostel", "Classroom", "Laboratory", "Campus Building"],
        "locations": [
            "Hostel Block A",
            "Hostel Block B",
            "C-204",
            "C-305",
            "Computer Laboratory",
            "Engineering Block",
        ],
        "issues_en": [
            "the ceiling fan is not working",
            "there is a power outage",
            "the tube light is flickering",
            "the electrical switch is damaged",
            "the corridor light is not working",
        ],
        "issues_ta_en": [
            "fan work aagala",
            "current supply illa",
            "tube light flicker aagudhu",
            "switch damage aagiruku",
            "corridor light eriyala",
        ],
        "issues_ta": [
            "மின்விசிறி வேலை செய்யவில்லை",
            "மின்சாரம் இல்லை",
            "குழல் விளக்கு மின்னுகிறது",
            "மின்சார சுவிட்ச் சேதமடைந்துள்ளது",
        ],
        "priority_weights": [0.20, 0.45, 0.28, 0.07],
        "resolution_range": (4, 30),
    },
    "Water and Plumbing": {
        "department": "Plumbing and Civil Maintenance",
        "location_types": ["Hostel", "Washroom", "Canteen", "Campus Building"],
        "locations": [
            "Hostel Block A",
            "Hostel Block B",
            "Hostel Block C",
            "Boys Hostel Washroom",
            "Girls Hostel Washroom",
            "Main Canteen",
            "Engineering Block",
        ],
        "issues_en": [
            "there is no water supply",
            "water is leaking from the pipe",
            "the washroom tap is broken",
            "the toilet is blocked",
            "the water tank is overflowing",
        ],
        "issues_ta_en": [
            "water supply varala",
            "pipe la water leak aagudhu",
            "washroom tap broken ah iruku",
            "toilet block aagiruku",
            "water tank overflow aagudhu",
        ],
        "issues_ta": [
            "தண்ணீர் வழங்கல் இல்லை",
            "குழாயில் தண்ணீர் கசிகிறது",
            "கழிவறை குழாய் உடைந்துள்ளது",
            "கழிவறை அடைத்துள்ளது",
        ],
        "priority_weights": [0.10, 0.38, 0.40, 0.12],
        "resolution_range": (3, 28),
    },
    "Hostel and Accommodation": {
        "department": "Hostel Administration",
        "location_types": ["Hostel"],
        "locations": [
            "Boys Hostel Block A",
            "Boys Hostel Block B",
            "Girls Hostel Block A",
            "Girls Hostel Block B",
            "Hostel Room 214",
            "Hostel Room 312",
        ],
        "issues_en": [
            "the room door lock is broken",
            "the room has a damaged window",
            "the hostel room is not cleaned",
            "the mattress is damaged",
            "the cupboard lock is not working",
        ],
        "issues_ta_en": [
            "room door lock broken ah iruku",
            "window damage aagiruku",
            "room clean pannala",
            "mattress damage aagiruku",
            "cupboard lock work aagala",
        ],
        "issues_ta": [
            "அறை கதவு பூட்டு உடைந்துள்ளது",
            "ஜன்னல் சேதமடைந்துள்ளது",
            "அறை சுத்தம் செய்யப்படவில்லை",
            "மெத்தை சேதமடைந்துள்ளது",
        ],
        "priority_weights": [0.22, 0.45, 0.28, 0.05],
        "resolution_range": (8, 72),
    },
    "Wi-Fi and IT Services": {
        "department": "IT Support and Network Cell",
        "location_types": ["Hostel", "Library", "Laboratory", "Classroom"],
        "locations": [
            "Central Library",
            "Central Library Reading Hall",
            "Computer Laboratory",
            "AI Laboratory",
            "Hostel Block A",
            "Hostel Block B",
        ],
        "issues_en": [
            "the Wi-Fi is not working",
            "the internet connection is very slow",
            "the student portal is not loading",
            "the ERP login is failing",
            "the laboratory computer cannot access the network",
        ],
        "issues_ta_en": [
            "wifi work aagala",
            "internet romba slow ah iruku",
            "student portal load aagala",
            "ERP login aagala",
            "lab computer network connect aagala",
        ],
        "issues_ta": [
            "வைஃபை வேலை செய்யவில்லை",
            "இணைய இணைப்பு மிகவும் மெதுவாக உள்ளது",
            "மாணவர் தளம் திறக்கவில்லை",
            "ஈஆர்பி உள்நுழைவு தோல்வியடைந்தது",
        ],
        "priority_weights": [0.18, 0.42, 0.33, 0.07],
        "resolution_range": (2, 36),
    },
    "Classroom and Laboratory": {
        "department": "Academic Infrastructure",
        "location_types": ["Classroom", "Laboratory"],
        "locations": [
            "C-101",
            "C-204",
            "C-305",
            "AI Laboratory",
            "Computer Laboratory",
            "Electronics Laboratory",
        ],
        "issues_en": [
            "the projector is not displaying",
            "the classroom bench is broken",
            "the laboratory equipment is not functioning",
            "the air conditioner is not working",
            "the classroom has poor ventilation",
        ],
        "issues_ta_en": [
            "projector display varala",
            "classroom bench broken ah iruku",
            "lab equipment work aagala",
            "AC work aagala",
            "classroom ventilation poor ah iruku",
        ],
        "issues_ta": [
            "ப்ரொஜெக்டர் காட்சி அளிக்கவில்லை",
            "வகுப்பறை மேசை உடைந்துள்ளது",
            "ஆய்வக உபகரணம் வேலை செய்யவில்லை",
            "குளிர்சாதன வசதி வேலை செய்யவில்லை",
        ],
        "priority_weights": [0.18, 0.47, 0.29, 0.06],
        "resolution_range": (4, 48),
    },
    "Cleanliness and Waste": {
        "department": "Housekeeping and Sanitation",
        "location_types": ["Hostel", "Washroom", "Canteen", "Campus Road"],
        "locations": [
            "Boys Hostel Washroom",
            "Girls Hostel Washroom",
            "Main Canteen",
            "Library Washroom",
            "Parking Area",
            "Academic Block Corridor",
        ],
        "issues_en": [
            "the washroom has not been cleaned",
            "garbage has accumulated",
            "the corridor is dirty",
            "there is a bad smell near the waste area",
            "the dustbin is overflowing",
        ],
        "issues_ta_en": [
            "washroom clean pannala",
            "garbage accumulate aagiruku",
            "corridor dirty ah iruku",
            "waste area pakkathula bad smell iruku",
            "dustbin overflow aagudhu",
        ],
        "issues_ta": [
            "கழிவறை சுத்தம் செய்யப்படவில்லை",
            "குப்பை குவிந்துள்ளது",
            "நடைபாதை அழுக்காக உள்ளது",
            "குப்பை இடத்தின் அருகில் துர்நாற்றம் உள்ளது",
        ],
        "priority_weights": [0.20, 0.45, 0.28, 0.07],
        "resolution_range": (2, 24),
    },
    "Safety and Security": {
        "department": "Security Office",
        "location_types": ["Hostel", "Campus Building", "Campus Road", "Parking"],
        "locations": [
            "Engineering Block",
            "Main Gate",
            "Parking Area",
            "Hostel Block B",
            "Campus Road",
            "Library Entrance",
        ],
        "issues_en": [
            "water is leaking near an electrical panel",
            "the staircase railing is loose",
            "there is a broken glass panel",
            "the campus street light is not working",
            "an unauthorized person entered the hostel area",
        ],
        "issues_ta_en": [
            "electrical panel pakkathula water leak aagudhu",
            "staircase railing loose ah iruku",
            "glass panel broken ah iruku",
            "campus street light work aagala",
            "unauthorized person hostel area la vandhirukanga",
        ],
        "issues_ta": [
            "மின்பலகையின் அருகில் தண்ணீர் கசிகிறது",
            "படிக்கட்டு கைப்பிடி தளர்வாக உள்ளது",
            "கண்ணாடி பலகம் உடைந்துள்ளது",
            "வளாக தெருவிளக்கு வேலை செய்யவில்லை",
        ],
        "priority_weights": [0.05, 0.18, 0.40, 0.37],
        "resolution_range": (1, 18),
    },
    "Transport and Parking": {
        "department": "Transport Cell",
        "location_types": ["Campus Road", "Parking"],
        "locations": [
            "Route 1 Bus Stop",
            "Route 2 Bus Stop",
            "Route 3 Bus Stop",
            "Main Parking Area",
            "College Entrance",
        ],
        "issues_en": [
            "the campus bus arrived late",
            "the bus route was changed without notice",
            "there is no parking space available",
            "a vehicle is blocking the campus entrance",
            "the bus is overcrowded",
        ],
        "issues_ta_en": [
            "campus bus late ah vandhuchu",
            "bus route notice illama change pannitanga",
            "parking space illa",
            "vehicle campus entrance block pannudhu",
            "bus romba crowded ah iruku",
        ],
        "issues_ta": [
            "வளாக பேருந்து தாமதமாக வந்தது",
            "அறிவிப்பின்றி பேருந்து பாதை மாற்றப்பட்டது",
            "நிறுத்துமிடம் கிடைக்கவில்லை",
            "வாகனம் வளாக நுழைவாயிலை மறைத்துள்ளது",
        ],
        "priority_weights": [0.22, 0.48, 0.25, 0.05],
        "resolution_range": (4, 40),
    },
    "Administration and Documents": {
        "department": "Administrative Office",
        "location_types": ["Office"],
        "locations": [
            "Academic Office",
            "Accounts Office",
            "Scholarship Office",
            "Examination Cell",
            "Student Service Centre",
        ],
        "issues_en": [
            "the bonafide certificate request is pending",
            "the fee payment is not updated in the portal",
            "the ID card request is delayed",
            "the scholarship application status is not updated",
            "the marksheet correction request is pending",
        ],
        "issues_ta_en": [
            "bonafide certificate request pending la iruku",
            "fee payment portal la update aagala",
            "ID card request delay aagudhu",
            "scholarship status update aagala",
            "marksheet correction pending la iruku",
        ],
        "issues_ta": [
            "பொனஃபைடு சான்றிதழ் கோரிக்கை நிலுவையில் உள்ளது",
            "கட்டண செலுத்துதல் தளத்தில் புதுப்பிக்கப்படவில்லை",
            "அடையாள அட்டை கோரிக்கை தாமதமாகிறது",
            "உதவித்தொகை நிலை புதுப்பிக்கப்படவில்லை",
        ],
        "priority_weights": [0.45, 0.40, 0.13, 0.02],
        "resolution_range": (24, 168),
    },
    "Food and Canteen": {
        "department": "Canteen Committee",
        "location_types": ["Canteen"],
        "locations": ["Main Canteen", "Hostel Mess", "North Canteen"],
        "issues_en": [
            "the food quality is poor",
            "the food appears unhygienic",
            "the canteen service is very slow",
            "the food price is incorrect",
            "the drinking water dispenser is not working",
        ],
        "issues_ta_en": [
            "food quality poor ah iruku",
            "food hygiene illa",
            "canteen service romba slow ah iruku",
            "food price wrong ah iruku",
            "drinking water dispenser work aagala",
        ],
        "issues_ta": [
            "உணவின் தரம் மோசமாக உள்ளது",
            "உணவு சுகாதாரமற்றதாக உள்ளது",
            "கேன்டீன் சேவை மிகவும் மெதுவாக உள்ளது",
            "உணவின் விலை தவறாக உள்ளது",
        ],
        "priority_weights": [0.18, 0.42, 0.31, 0.09],
        "resolution_range": (2, 30),
    },
}

PRIORITIES = ["Low", "Medium", "High", "Critical"]
LANGUAGES = ["en", "ta_en", "ta"]
LANGUAGE_WEIGHTS = [0.45, 0.30, 0.25]
STATUSES = ["Open", "In Progress", "Resolved", "Closed"]

EN_TEMPLATES = [
    "Please check {issue} at {location}.",
    "I want to report that {issue} in {location}.",
    "There is a problem because {issue} at {location}.",
    "Urgent attention is needed because {issue} in {location}.",
]

TA_EN_TEMPLATES = [
    "{location} la {issue}. Please check pannunga.",
    "{location} la {issue}, immediate ah solve pannunga.",
    "{issue} at {location}. Kindly action edunga.",
    "{location} area la {issue}.",
]

TA_TEMPLATES = [
    "{location} பகுதியில் {issue}. தயவுசெய்து சரிசெய்யவும்.",
    "{location} இல் {issue}. உடனடியாக நடவடிக்கை எடுக்கவும்.",
    "{issue} பிரச்சனை {location} இடத்தில் உள்ளது.",
]


def normalize_text(text: str) -> str:
    return " ".join(
        "".join(char.lower() if char.isalnum() or char.isspace() else " " for char in text).split()
    )


def choose_status() -> str:
    return random.choices(STATUSES, weights=[0.22, 0.22, 0.38, 0.18], k=1)[0]


def generate_text(language: str, issue: str, location: str) -> str:
    if language == "en":
        return random.choice(EN_TEMPLATES).format(issue=issue, location=location)
    if language == "ta_en":
        return random.choice(TA_EN_TEMPLATES).format(issue=issue, location=location)
    return random.choice(TA_TEMPLATES).format(issue=issue, location=location)


def select_affected_population(location_type: str) -> int:
    ranges = {
        "Hostel": (20, 220),
        "Classroom": (25, 75),
        "Laboratory": (20, 80),
        "Library": (40, 180),
        "Washroom": (15, 120),
        "Canteen": (30, 160),
        "Campus Building": (50, 250),
        "Campus Road": (25, 180),
        "Parking": (10, 120),
        "Office": (1, 25),
    }
    low, high = ranges.get(location_type, (10, 100))
    return random.randint(low, high)


def calculate_resolution_time(
    priority: str,
    resolution_range: tuple[int, int],
    repeat_count: int,
    safety_flag: int,
) -> float:
    low, high = resolution_range
    base = random.uniform(low, high)

    priority_factor = {
        "Low": 1.30,
        "Medium": 1.00,
        "High": 0.72,
        "Critical": 0.35,
    }[priority]

    repeat_factor = max(0.70, 1 - (repeat_count * 0.05))
    safety_factor = 0.75 if safety_flag else 1.0
    noise = random.uniform(0.80, 1.20)

    return round(max(0.5, base * priority_factor * repeat_factor * safety_factor * noise), 2)


def make_record(
    complaint_id: str,
    category: str,
    issue_index: int,
    language: str,
    location: str,
    location_type: str,
    priority: str,
    created_at: datetime,
    repeat_count: int,
    is_duplicate: int = 0,
    duplicate_of_id: str | None = None,
) -> dict:
    config = CATEGORY_CONFIG[category]
    issue_key = {
        "en": "issues_en",
        "ta_en": "issues_ta_en",
        "ta": "issues_ta",
    }[language]

    issue = config[issue_key][issue_index % len(config[issue_key])]
    complaint_text = generate_text(language, issue, location)
    safety_flag = int(
        category == "Safety and Security"
        or (category == "Food and Canteen" and priority in {"High", "Critical"})
        or (category == "Electrical and Power" and priority == "Critical")
    )

    affected_population = select_affected_population(location_type)
    resolution_time_hours = calculate_resolution_time(
        priority=priority,
        resolution_range=config["resolution_range"],
        repeat_count=repeat_count,
        safety_flag=safety_flag,
    )

    status = choose_status()
    notes = "" if status in {"Open", "In Progress"} else "Synthetic resolution record"

    return {
        "complaint_id": complaint_id,
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "language": language,
        "complaint_text": complaint_text,
        "normalized_text": normalize_text(complaint_text),
        "category": category,
        "department": config["department"],
        "priority": priority,
        "location_type": location_type,
        "specific_location": location,
        "affected_population": affected_population,
        "safety_flag": safety_flag,
        "repeat_count": repeat_count,
        "is_duplicate": is_duplicate,
        "duplicate_of_id": duplicate_of_id,
        "resolution_time_hours": resolution_time_hours,
        "status": status,
        "resolution_notes": notes,
    }


def build_dataset() -> pd.DataFrame:
    records: list[dict] = []
    categories = list(CATEGORY_CONFIG)
    start_date = datetime(2026, 1, 1, 8, 0, 0)

    for index in range(1, TOTAL_UNIQUE_COMPLAINTS + 1):
        category = random.choice(categories)
        config = CATEGORY_CONFIG[category]
        language = random.choices(LANGUAGES, weights=LANGUAGE_WEIGHTS, k=1)[0]
        issue_index = next(
            i
            for i, issue in enumerate(config["issues_en"])
            if issue in original["complaint_text"].lower()
        ) if original["language"] == "en" else 0
        location_type = random.choice(config["location_types"])
        location = random.choice(config["locations"])
        priority = random.choices(PRIORITIES, weights=config["priority_weights"], k=1)[0]
        created_at = start_date + timedelta(
            days=random.randint(0, 240),
            hours=random.randint(0, 12),
            minutes=random.randint(0, 59),
        )
        repeat_count = random.randint(0, 2)
        complaint_id = f"CMP-{index:05d}"

        original_record = make_record(
            complaint_id=complaint_id,
            category=category,
            issue_index=issue_index,
            language=language,
            location=location,
            location_type=location_type,
            priority=priority,
            created_at=created_at,
            repeat_count=repeat_count,
        )
        records.append(original_record)

    duplicate_count = int(TOTAL_UNIQUE_COMPLAINTS * DUPLICATE_RATE)
    originals_for_duplicates = random.sample(records, duplicate_count)

    for offset, original in enumerate(originals_for_duplicates, start=1):
        category = original["category"]
        config = CATEGORY_CONFIG[category]
        original_language = original["language"]
        alternate_languages = [lang for lang in LANGUAGES if lang != original_language]
        duplicate_language = random.choice(alternate_languages)
        issue_index = random.randrange(len(config["issues_en"]))

        duplicate_id = f"CMP-{TOTAL_UNIQUE_COMPLAINTS + offset:05d}"
        original_time = datetime.strptime(original["created_at"], "%Y-%m-%d %H:%M:%S")
        duplicate_time = original_time + timedelta(
            hours=random.randint(1, 72),
            minutes=random.randint(0, 59),
        )

        duplicate_record = make_record(
            complaint_id=duplicate_id,
            category=category,
            issue_index=issue_index,
            language=duplicate_language,
            location=original["specific_location"],
            location_type=original["location_type"],
            priority=original["priority"],
            created_at=duplicate_time,
            repeat_count=int(original["repeat_count"]) + 1,
            is_duplicate=1,
            duplicate_of_id=original["complaint_id"],
        )
        records.append(duplicate_record)

    df = pd.DataFrame(records)
    return df.sort_values("created_at").reset_index(drop=True)


def validate_generated_data(df: pd.DataFrame) -> None:
    expected_total = TOTAL_UNIQUE_COMPLAINTS + int(TOTAL_UNIQUE_COMPLAINTS * DUPLICATE_RATE)

    assert len(df) == expected_total, f"Expected {expected_total} rows, found {len(df)}"
    assert df["complaint_id"].is_unique, "Complaint IDs must be unique"
    assert df["complaint_text"].notna().all(), "Complaint text cannot be null"
    assert df["category"].nunique() == len(CATEGORY_CONFIG), "All categories must be present"
    assert set(df["language"]).issubset(set(LANGUAGES)), "Unexpected language label"
    assert set(df["priority"]).issubset(set(PRIORITIES)), "Unexpected priority label"
    assert (df["resolution_time_hours"] > 0).all(), "Resolution time must be positive"

    duplicates = df[df["is_duplicate"] == 1]
    assert len(duplicates) == int(TOTAL_UNIQUE_COMPLAINTS * DUPLICATE_RATE)
    assert duplicates["duplicate_of_id"].notna().all(), "Duplicate records need original IDs"

    original_ids = set(df["complaint_id"])
    assert set(duplicates["duplicate_of_id"]).issubset(original_ids), (
        "Every duplicate reference must point to an existing complaint"
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    validate_generated_data(df)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Dataset generated successfully: {OUTPUT_PATH}")
    print(f"Total records: {len(df)}")
    print(f"Original complaints: {(df['is_duplicate'] == 0).sum()}")
    print(f"Duplicate complaints: {(df['is_duplicate'] == 1).sum()}")

    print("\nCategory distribution:")
    print(df["category"].value_counts().sort_index().to_string())

    print("\nPriority distribution:")
    print(df["priority"].value_counts().to_string())

    print("\nLanguage distribution:")
    print(df["language"].value_counts().to_string())


if __name__ == "__main__":
    main()