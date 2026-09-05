{
  "scenario_type": "expense fraud",
  "persona": "Senior Corporate Fraud Auditor, sharp and methodical",
  "guilty": true,
  "arrest_threshold": 80,
  "policy_rules": [
    "Itemized Receipt Mandatory: Any corporate card charge over $100 requires an itemized receipt showing line items.",
    "24-Hour Reporting Policy: Any suspected billing error, unauthorized charge, or material transaction issue over $200 must be reported to Finance or the card issuer within 24 hours.",
    "Individual Meal Cap: Non-client individual meals on business trips are capped at $75 USD per day unless an approved exception exists."
  ],
  "facts": [
    {
      "id": "solo_dinner",
      "description": "Suspect dined alone; reservation was for one person",
      "true_value": "true",
      "weight": "high",
      "certainty": "fast",
      "visible_to": [
        "suspect",
        "investigator_start"
      ]
    },
    {
      "id": "no_client_meeting",
      "description": "No client meeting occurred that night",
      "true_value": "true",
      "weight": "high",
      "certainty": "fast",
      "visible_to": [
        "suspect"
      ]
    },
    {
      "id": "no_calendar_entry",
      "description": "No calendar entry exists for that night",
      "true_value": "true",
      "weight": "medium",
      "certainty": "fast",
      "visible_to": [
        "suspect",
        "investigator_start"
      ]
    },
    {
      "id": "actual_amount",
      "description": "The corporate card transaction settled to the restaurant",
      "true_value": "$3,200",
      "weight": "high",
      "certainty": "fast",
      "visible_to": [
        "suspect",
        "investigator_start"
      ]
    },
    {
      "id": "no_supporting_documentation",
      "description": "No supporting documentation or approval explaining the exceptional expense was submitted with the transaction",
      "true_value": "true",
      "weight": "high",
      "certainty": "fast",
      "visible_to": [
        "investigator_start"
      ]
    },
    {
      "id": "no_prior_issue_report",
      "description": "No issue concerning the transaction was reported through company or card channels before the investigation began",
      "true_value": "true",
      "weight": "high",
      "certainty": "fast",
      "visible_to": [
        "investigator_start"
      ]
    },
    {
      "id": "standard_record_review",
      "description": "The records available before the interview contain no documented explanation that resolves why the transaction differs from normal policy expectations",
      "true_value": "true",
      "weight": "medium",
      "certainty": "fast",
      "visible_to": [
        "investigator_start"
      ]
    }
  ],
  "claim_categories": []
}