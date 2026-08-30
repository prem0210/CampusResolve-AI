# CampusResolve-AI Dataset Dictionary

## Dataset purpose

This dataset supports multilingual complaint classification, department routing,
duplicate detection, priority forecasting, and resolution-time prediction for
campus facilities and services.

## Columns

| Column | Type | Description | Example |
|---|---|---|---|
| complaint_id | string | Unique complaint identifier | CMP-0001 |
| created_at | datetime | Complaint creation timestamp in ISO 8601 format | 2026-08-24 09:30:00 |
| language | category | Complaint language: en, ta, or ta_en | ta_en |
| complaint_text | text | Original complaint submitted by a student | Hostel Block B la water varala |
| normalized_text | text | Cleaned text for ML preprocessing | hostel block b la water varala |
| category | category | One of the 10 complaint categories | Water and Plumbing |
| department | category | Department assigned to resolve the complaint | Plumbing and Civil Maintenance |
| priority | category | Operational priority: Low, Medium, High, Critical | High |
| location_type | category | Broad location of the reported issue | Hostel |
| specific_location | string | Specific campus location or building | Hostel Block B |
| affected_population | integer | Approximate number of people affected | 150 |
| safety_flag | integer | 1 if the complaint has a safety concern; otherwise 0 | 1 |
| repeat_count | integer | Number of earlier reports for the same issue/location | 2 |
| is_duplicate | integer | 1 if record duplicates a previous complaint; otherwise 0 | 0 |
| duplicate_of_id | string/null | Original complaint ID if duplicate; blank otherwise | CMP-0001 |
| resolution_time_hours | float | Actual or synthetic time required for resolution | 10.5 |
| status | category | Complaint state: Open, In Progress, Resolved, Closed | Open |
| resolution_notes | text/null | Resolution details, if available | Plumber assigned |

## Complaint categories and departments

| Category | Department |
|---|---|
| Electrical and Power | Electrical Maintenance |
| Water and Plumbing | Plumbing and Civil Maintenance |
| Hostel and Accommodation | Hostel Administration |
| Wi-Fi and IT Services | IT Support and Network Cell |
| Classroom and Laboratory | Academic Infrastructure |
| Cleanliness and Waste | Housekeeping and Sanitation |
| Safety and Security | Security Office |
| Transport and Parking | Transport Cell |
| Administration and Documents | Administrative Office |
| Food and Canteen | Canteen Committee |

## Priority definition

| Priority | Definition |
|---|---|
| Low | Minor inconvenience without immediate service disruption or safety concern |
| Medium | Affects an individual or small group and needs routine attention |
| High | Disrupts services, learning, accommodation, sanitation, or affects many people |
| Critical | Immediate safety, health, security, fire, electrical, or severe infrastructure risk |

## Ethical and privacy notes

- Do not include real student names, phone numbers, roll numbers, email addresses, or personally identifiable information.
- Do not generate abusive, defamatory, or personally identifying complaint content.
- Safety and security complaints are decision-support outputs only; campus staff must make final escalation decisions.
- Synthetic records must be clearly documented as synthetic when used in reports or publications.