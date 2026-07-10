"""QA dataset over the AcmeCloud knowledge base (data/acmecloud_kb/).

100 cases, 25 per difficulty tier:

- easy    — single fact, phrased close to the source wording;
- medium  — single fact, paraphrased, so lexical overlap with the source drops;
- hard    — synthesis: comparing plans, combining facts, or light arithmetic;
- expert  — cross-document reasoning, plus 8 *unanswerable* questions that a
            grounded system must abstain on instead of hallucinating.

Each case carries:
- ``accepted``     — strings, any of which makes a prediction correct (purely
                     numeric strings are matched with digit boundaries so "50"
                     cannot be satisfied by "500");
- ``sources``      — gold doc_ids (file stems in data/acmecloud_kb/) used to
                     score retrieval recall; empty for unanswerable cases;
- ``derived``      — True when the answer is computed from the docs rather than
                     quoted, so the "answer appears in a gold doc" integrity
                     test skips it;
- ``unanswerable`` — True when the correct behaviour is to abstain.

Tests enforce that every answerable, non-derived case's accepted answer really
occurs in one of its gold source documents.
"""

from typing import Any, Dict, List

# Answers accepted as a correct abstention (unanswerable cases). The RAG prompt
# asks for the exact phrase "Not in the documentation"; the baseline prompt asks
# for "I don't know"; the rest catch common paraphrases.
ABSTAIN_ANSWERS = [
    "not in the documentation",
    "not in the docs",
    "i don't know",
    "i do not know",
    "not mentioned",
    "not specified",
    "not documented",
    "does not specify",
    "no information",
]


def _c(
    question: str,
    accepted: List[str],
    sources: List[str],
    difficulty: str,
    derived: bool = False,
    unanswerable: bool = False,
) -> Dict[str, Any]:
    return {
        "question": question,
        "accepted": ABSTAIN_ANSWERS if unanswerable else accepted,
        "sources": sources,
        "difficulty": difficulty,
        "derived": derived,
        "unanswerable": unanswerable,
    }


EASY: List[Dict[str, Any]] = [
    _c(
        "How many concurrent pipelines does the AcmeCloud Growth plan allow?",
        ["50"],
        ["plans_and_billing"],
        "easy",
    ),
    _c(
        "How much managed storage does the Starter plan include?",
        ["50 GB", "50GB"],
        ["plans_and_billing"],
        "easy",
    ),
    _c(
        "After how many days does a default AcmeCloud AKT token expire?",
        ["14"],
        ["authentication"],
        "easy",
    ),
    _c(
        "Which command rotates an AKT without downtime?",
        ["acme auth rotate"],
        ["authentication", "cli_reference"],
        "easy",
    ),
    _c(
        "What is the default region for new AcmeCloud accounts created after January 2025?",
        ["eu-paris-1"],
        ["regions_and_infrastructure"],
        "easy",
    ),
    _c(
        "How many free ACUs per month does the Growth plan include?",
        ["500"],
        ["plans_and_billing"],
        "easy",
    ),
    _c(
        "How much is each additional ACU billed at on the Growth plan?",
        ["0.08"],
        ["plans_and_billing"],
        "easy",
    ),
    _c(
        "On which day of the month are AcmeCloud invoices issued?",
        ["3rd", "third"],
        ["plans_and_billing"],
        "easy",
    ),
    _c(
        "What is the Enterprise severity-1 support response time?",
        ["1 hour", "1-hour", "one hour"],
        ["support_sla"],
        "easy",
    ),
    _c(
        "How long are pipeline logs retained on the Enterprise plan?",
        ["400"],
        ["data_retention"],
        "easy",
    ),
    _c(
        "What minimum Python version does the acmecloud-sdk require?",
        ["3.10"],
        ["sdk_python"],
        "easy",
    ),
    _c(
        "What is the maximum webhook payload size in AcmeCloud?",
        ["256"],
        ["api_reference"],
        "easy",
    ),
    _c(
        "How long do deleted datasets stay recoverable in the trash?",
        ["21"],
        ["data_retention"],
        "easy",
    ),
    _c(
        "In how many regions is AcmeCloud available?",
        ["three", "3"],
        ["regions_and_infrastructure"],
        "easy",
    ),
    _c(
        "What is the API rate limit on the Starter plan?",
        ["600"],
        ["plans_and_billing", "api_reference"],
        "easy",
    ),
    _c(
        "What kind of token do AcmeCloud service accounts use?",
        ["AKT-S"],
        ["authentication"],
        "easy",
    ),
    _c(
        "What is the monthly price of the Starter plan?",
        ["49"],
        ["plans_and_billing"],
        "easy",
    ),
    _c(
        "What is the codename of AcmeCloud release v3.2?",
        ["Meridian"],
        ["overview", "release_notes"],
        "easy",
    ),
    _c(
        "Which encryption standard protects AcmeCloud data at rest?",
        ["AES-256"],
        ["security_compliance"],
        "easy",
    ),
    _c(
        "To which address should security vulnerabilities be reported?",
        ["security@acmecloud.example"],
        ["security_compliance"],
        "easy",
    ),
    _c(
        "What is the escalation contact for severity-1 Enterprise incidents?",
        ["sev1@acmecloud.example"],
        ["support_sla"],
        "easy",
    ),
    _c(
        "What is the base URL of the AcmeCloud REST API?",
        ["api.acmecloud.example/v3"],
        ["api_reference"],
        "easy",
    ),
    _c(
        "What is the default page size for API list endpoints?",
        ["50"],
        ["api_reference"],
        "easy",
    ),
    _c(
        "What is the address of the AcmeCloud service-status page?",
        ["status.acmecloud.example"],
        ["overview", "troubleshooting"],
        "easy",
    ),
    _c(
        "How many times is a webhook delivery attempted?",
        ["5", "five"],
        ["api_reference"],
        "easy",
    ),
]

MEDIUM: List[Dict[str, Any]] = [
    _c(
        "A customer insists on paying by wire transfer. Which plan supports that?",
        ["Enterprise"],
        ["plans_and_billing"],
        "medium",
    ),
    _c(
        "My token starts with the prefix akts_. After how many days will it expire?",
        ["90"],
        ["authentication"],
        "medium",
    ),
    _c(
        "For how long is an account locked after too many failed login attempts?",
        ["15"],
        ["authentication"],
        "medium",
    ),
    _c(
        "Which single sign-on protocol is already available on the Growth plan?",
        ["SAML"],
        ["authentication"],
        "medium",
    ),
    _c(
        "Which AcmeCloud region came online in October 2024?",
        ["ap-osaka-3"],
        ["regions_and_infrastructure"],
        "medium",
    ),
    _c(
        "Between which two regions is cross-region replication supported?",
        ["eu-paris-1 and us-reston-2", "us-reston-2 and eu-paris-1"],
        ["regions_and_infrastructure"],
        "medium",
    ),
    _c(
        "What discount does annual billing give on AcmeCloud plans?",
        ["15%", "15 percent"],
        ["plans_and_billing"],
        "medium",
    ),
    _c(
        "How long does the AcmeCloud free trial last?",
        ["30"],
        ["plans_and_billing"],
        "medium",
    ),
    _c(
        "When does a plan downgrade take effect?",
        ["next billing cycle"],
        ["plans_and_billing"],
        "medium",
    ),
    _c(
        "How many availability zones does each region span at minimum?",
        ["3", "three"],
        ["regions_and_infrastructure"],
        "medium",
    ),
    _c(
        "What is the maximum lifetime an AKT can be configured to have?",
        ["60"],
        ["authentication"],
        "medium",
    ),
    _c(
        "Which HTTP status code does the API return when the rate limit is exceeded?",
        ["429"],
        ["api_reference"],
        "medium",
    ),
    _c(
        "For how long is an Idempotency-Key valid after first use?",
        ["24"],
        ["api_reference"],
        "medium",
    ),
    _c(
        "What is the maximum page_size accepted by list endpoints?",
        ["200"],
        ["api_reference"],
        "medium",
    ),
    _c(
        "Which file does the acme CLI read its configuration from?",
        ["config.toml"],
        ["cli_reference"],
        "medium",
    ),
    _c(
        "Which command generates a diagnostics bundle to attach to a support ticket?",
        ["doctor --bundle"],
        ["cli_reference", "troubleshooting"],
        "medium",
    ),
    _c(
        "What is the recommended way to install the acme CLI?",
        ["pipx"],
        ["cli_reference"],
        "medium",
    ),
    _c(
        "How many times does the Python SDK retry a failed request by default?",
        ["3", "three"],
        ["sdk_python"],
        "medium",
    ),
    _c(
        "What is the default request timeout of the Python SDK client?",
        ["30"],
        ["sdk_python"],
        "medium",
    ),
    _c(
        "Since which SDK version is async support available?",
        ["2.5"],
        ["sdk_python"],
        "medium",
    ),
    _c(
        "How often are backup encryption keys rotated?",
        ["90"],
        ["security_compliance"],
        "medium",
    ),
    _c(
        "For how many days are audit logs retained?",
        ["365"],
        ["security_compliance"],
        "medium",
    ),
    _c(
        "What is the monthly uptime SLA of the Growth plan?",
        ["99.9"],
        ["support_sla"],
        "medium",
    ),
    _c(
        "On which day of the week do scheduled maintenance windows happen?",
        ["Sunday"],
        ["support_sla"],
        "medium",
    ),
    _c(
        "For how many days are automated backups retained?",
        ["35"],
        ["data_retention"],
        "medium",
    ),
]

HARD: List[Dict[str, Any]] = [
    _c(
        "How many more concurrent pipelines does the Growth plan allow compared to Starter?",
        ["45"],
        ["plans_and_billing"],
        "hard",
        derived=True,
    ),
    _c(
        "For how long is dataset version history kept on the Growth plan?",
        ["90"],
        ["data_retention"],
        "hard",
    ),
    _c(
        "A pipeline has been stuck in the PENDING state for an hour. What is the most likely cause?",
        ["concurrent", "quota"],
        ["troubleshooting"],
        "hard",
    ),
    _c(
        "A request fails with error code ACME-4031. What does it mean?",
        ["expired"],
        ["api_reference", "troubleshooting"],
        "hard",
    ),
    _c(
        "How many more days of pipeline log retention does Enterprise offer over Growth?",
        ["220"],
        ["data_retention"],
        "hard",
        derived=True,
    ),
    _c(
        "A Starter customer used 150 ACUs this month. How much will the extra ACUs cost?",
        ["$6", "6.00", "6 dollars", "6 USD"],
        ["plans_and_billing"],
        "hard",
        derived=True,
    ),
    _c(
        "What is the cheapest plan that includes live chat support?",
        ["Growth"],
        ["support_sla"],
        "hard",
    ),
    _c(
        "Where can Enterprise customers export their audit logs?",
        ["S3"],
        ["security_compliance"],
        "hard",
    ),
    _c(
        "Up to how many hours can the maximum pipeline runtime be raised on Enterprise?",
        ["72"],
        ["troubleshooting"],
        "hard",
    ),
    _c(
        "A 20 GB dataset file fails to upload through the web console. What should the customer use instead?",
        ["CLI", "multipart"],
        ["troubleshooting"],
        "hard",
    ),
    _c(
        "For how many seconds does the API tolerate bursts above the plan rate limit?",
        ["60"],
        ["api_reference"],
        "hard",
    ),
    _c(
        "What is the burst ceiling, in requests per minute, on the Growth plan?",
        ["6,000", "6000"],
        ["api_reference", "plans_and_billing"],
        "hard",
        derived=True,
    ),
    _c(
        "Which release made multi-factor authentication mandatory for admin accounts?",
        ["Lyra", "v3.1"],
        ["release_notes", "authentication"],
        "hard",
    ),
    _c(
        "Which deprecated Python package will release v4.0 remove?",
        ["acme-py"],
        ["release_notes", "sdk_python"],
        "hard",
    ),
    _c(
        "How many months of notice does AcmeCloud give before a breaking API change?",
        ["12", "twelve"],
        ["release_notes"],
        "hard",
    ),
    _c(
        "What is the severity-1 response time on the Growth plan?",
        ["4 hour", "4-hour", "four hour"],
        ["support_sla"],
        "hard",
    ),
    _c(
        "A Growth customer's uptime was 99.7% last month. What SLA credit do they get?",
        ["20%", "20 percent"],
        ["support_sla"],
        "hard",
        derived=True,
    ),
    _c(
        "Which plan includes point-in-time recovery, and how long is the window?",
        ["7-day", "7 day", "seven-day"],
        ["data_retention"],
        "hard",
    ),
    _c(
        "What was the default region for new accounts before January 2025?",
        ["us-reston-2"],
        ["regions_and_infrastructure"],
        "hard",
    ),
    _c(
        "Adding up all three plans, how many free ACUs per month do they include in total?",
        ["5,600", "5600"],
        ["plans_and_billing"],
        "hard",
        derived=True,
    ),
    _c(
        "What happens to a webhook event once all delivery retries have failed?",
        ["dropped"],
        ["troubleshooting", "api_reference"],
        "hard",
    ),
    _c(
        "Which orchestration component was introduced in release v3.0?",
        ["Flows"],
        ["release_notes", "overview"],
        "hard",
    ),
    _c(
        "How many times more managed storage does Enterprise include compared to Growth?",
        ["20 times", "20×", "20x", "twenty times"],
        ["plans_and_billing"],
        "hard",
        derived=True,
    ),
    _c(
        "What is the API rate limit on the plan with the highest limit?",
        ["20,000", "20000"],
        ["plans_and_billing", "api_reference"],
        "hard",
    ),
    _c(
        "Does data stored by EU customers in eu-paris-1 ever leave that region?",
        ["never leaves", "does not leave", "stays in"],
        ["regions_and_infrastructure"],
        "hard",
    ),
]

EXPERT: List[Dict[str, Any]] = [
    _c(
        "An admin account on release v3.2 has no MFA configured. Is that allowed?",
        ["mandatory", "not allowed", "required"],
        ["authentication", "release_notes"],
        "expert",
    ),
    _c(
        "A request fails with ACME-4031 and the token is prefixed akts_. How many days ago, "
        "at minimum, was that token issued?",
        ["90"],
        ["authentication", "api_reference"],
        "expert",
    ),
    _c(
        "A Growth customer wants OIDC single sign-on and SCIM provisioning. What do they need to do?",
        ["Enterprise", "upgrade"],
        ["authentication"],
        "expert",
    ),
    _c(
        "Which release introduced Flows, and what is that release's codename?",
        ["Kestrel"],
        ["release_notes"],
        "expert",
    ),
    _c(
        "A Growth customer consumed 800 ACUs this month. Including the plan fee, what is their "
        "total monthly bill in dollars?",
        ["523"],
        ["plans_and_billing"],
        "expert",
        derived=True,
    ),
    _c(
        "An Enterprise dataset was deleted 30 days ago, past the trash window. Which recovery "
        "option can still contain it?",
        ["backup"],
        ["data_retention"],
        "expert",
    ),
    _c(
        "Our webhook posts a 300 KB payload and deliveries never arrive. Why?",
        ["256"],
        ["api_reference", "troubleshooting"],
        "expert",
    ),
    _c(
        "A Starter client calls the API at 700 requests per minute continuously. What response "
        "will it start receiving after the 60-second burst window?",
        ["429"],
        ["api_reference"],
        "expert",
    ),
    _c(
        "Since what year has AcmeCloud been ISO 27001 certified?",
        ["2024"],
        ["security_compliance"],
        "expert",
    ),
    _c(
        "By when must customers migrate off the acme-py package?",
        ["March 2026"],
        ["release_notes", "sdk_python"],
        "expert",
    ),
    _c(
        "How many percentage points separate the Growth and Enterprise uptime SLAs?",
        ["0.05"],
        ["support_sla"],
        "expert",
        derived=True,
    ),
    _c(
        "An account was created in December 2024. Which region was it placed in by default?",
        ["us-reston-2"],
        ["regions_and_infrastructure"],
        "expert",
    ),
    _c(
        "What is the maximum SLA credit, as a share of the monthly fee, that a customer can receive?",
        ["30"],
        ["support_sla"],
        "expert",
    ),
    _c(
        "A Growth customer asks for pipeline logs from 200 days ago. Are they still available?",
        ["180", "export"],
        ["data_retention"],
        "expert",
    ),
    _c(
        "Which command does the troubleshooting guide recommend running before opening a "
        "connectivity support ticket?",
        ["doctor"],
        ["troubleshooting", "cli_reference"],
        "expert",
    ),
    _c(
        "Running one ACU's worth of compute for 3 hours corresponds to how many vCPU-hours?",
        ["12", "twelve"],
        ["plans_and_billing"],
        "expert",
        derived=True,
    ),
    _c(
        "A Starter customer hit their storage cap. How much managed storage does their plan include?",
        ["50 GB", "50GB"],
        ["plans_and_billing"],
        "expert",
    ),
    # --- Unanswerable: the docs say nothing about these. A grounded system must abstain. ---
    _c(
        "What is the maximum number of Flows allowed per project?",
        [],
        [],
        "expert",
        unanswerable=True,
    ),
    _c(
        "Does AcmeCloud provide an official Terraform provider?",
        [],
        [],
        "expert",
        unanswerable=True,
    ),
    _c(
        "Is AcmeCloud HIPAA compliant?",
        [],
        [],
        "expert",
        unanswerable=True,
    ),
    _c(
        "Which public cloud provider hosts AcmeCloud's Kubernetes clusters?",
        [],
        [],
        "expert",
        unanswerable=True,
    ),
    _c(
        "How long are Lenses dashboard snapshots retained?",
        [],
        [],
        "expert",
        unanswerable=True,
    ),
    _c(
        "How many engineers work on the AcmeCloud platform team?",
        [],
        [],
        "expert",
        unanswerable=True,
    ),
    _c(
        "What is the rate limit of the legacy REST v2 API?",
        [],
        [],
        "expert",
        unanswerable=True,
    ),
    _c(
        "Does the acme CLI support Windows?",
        [],
        [],
        "expert",
        unanswerable=True,
    ),
]

CASES: List[Dict[str, Any]] = EASY + MEDIUM + HARD + EXPERT

TIERS = ["easy", "medium", "hard", "expert"]
