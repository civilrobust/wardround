"""
MWIA v2 — Modern Ward Round Intelligence Advisor
=================================================
King's College Hospital NHS Foundation Trust
ICT AI Services — David, AI Engineer

Enhanced FastAPI backend with full ward round workflow:
  • Board Round mode (SORT: Sick/Out/Rest/To-come-in)
  • Tasks/Jobs system with ownership and status
  • Safety Bundles (VTE/ABX/Lines/DNACPR checklists)
  • MDT roles (nurse/pharmacy/therapy inputs)
  • Optimized CSV import with validation
  • OpenAI Voice (Whisper STT + TTS) + GPT-4o reasoning

Environment variables:
  OPENAI_API_KEY      — required for voice + AI
  MWIA_PASS_ADMIN     — admin password
  MWIA_PASS_DOCTOR    — doctor password
  MWIA_PASS_NURSE     — nurse password
  MWIA_PASS_DEMO      — demo password

Deploy on Linode:
  pip install fastapi uvicorn openai pandas python-multipart
  export OPENAI_API_KEY=sk-...
  uvicorn mwia_backend_v2:app --host 0.0.0.0 --port 8001 --reload

NOT FOR CLINICAL USE — Development Prototype
"""

import os, io, csv, uuid, hashlib, secrets, tempfile, re
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Response, Cookie
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client  = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TTS_VOICE  = os.getenv("TTS_VOICE",  "shimmer")
TTS_MODEL  = os.getenv("TTS_MODEL",  "tts-1")
AI_MODEL   = os.getenv("AI_MODEL",   "gpt-4o")
STT_MODEL  = os.getenv("STT_MODEL",  "whisper-1")

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def _h(s): return hashlib.sha256(s.encode()).hexdigest()

MWIA_USERS = {
    "admin":  _h(os.getenv("MWIA_PASS_ADMIN",  "kingsAI26")),
    "doctor": _h(os.getenv("MWIA_PASS_DOCTOR", "wardRound1")),
    "nurse":  _h(os.getenv("MWIA_PASS_NURSE",  "nurseKCH1")),
    "demo":   _h(os.getenv("MWIA_PASS_DEMO",   "demo2026!")),
}
active_sessions: Dict[str, str] = {}

def verify_login(user: str, pw: str) -> bool:
    return MWIA_USERS.get(user.lower()) == _h(pw)

def create_session(user: str) -> str:
    tok = secrets.token_hex(32)
    active_sessions[tok] = user
    return tok

def get_user(session_token: Optional[str] = Cookie(default=None)) -> Optional[str]:
    return active_sessions.get(session_token or "")

def require_user(session_token: Optional[str] = Cookie(default=None)) -> str:
    u = get_user(session_token)
    if not u:
        raise HTTPException(401, "Not authenticated")
    return u

# ─────────────────────────────────────────────
# IN-MEMORY STORES
# ─────────────────────────────────────────────
patients:      Dict[str, Dict] = {}   # pid -> patient dict
round_notes:   Dict[str, List] = {}   # pid -> list of note dicts
tasks:         Dict[str, Dict] = {}   # task_id -> task dict
safety_checks: Dict[str, Dict] = {}   # pid -> safety bundle dict
mdt_inputs:    Dict[str, Dict] = {}   # pid -> {nursing: ..., pharmacy: ..., therapy: ...}

# Board round session
board_round_session: Optional[Dict] = None  # {date, ward, team, attendees, started_at}

# ─────────────────────────────────────────────
# SCHEMA — WARD ROUND FIELDS
# ─────────────────────────────────────────────
WARD_FIELDS = [
    # Identity
    "patient_id","name","nhs_number","dob","age","gender",
    # Location
    "ward","bay","bed","hospital_site",
    # Clinical team
    "consultant","registrar","team","specialty",
    # Admission
    "admission_date","admission_reason","days_in","patient_class",
    # Priority / SORT
    "priority","sort_category","escalation_flag","dnacpr","isolation","vte_status",
    # DAVID & WENDY framework
    "dw_diet","dw_activity","dw_vitals","dw_investigations",
    "dw_drains","dw_wounds","dw_examination","dw_nursing","dw_drugs","dw_barriers",
    # Observations (NEWS2)
    "obs_hr","obs_hr_flag","obs_bp","obs_bp_flag",
    "obs_temp","obs_temp_flag","obs_o2","obs_o2_flag","obs_rr","obs_rr_flag",
    "obs_consciousness","obs_urine_output","obs_pain_score",
    "news_score","news_trend",
    # Medications
    "medications","allergies","antibiotics","abx_indication","abx_day","abx_review","abx_stop_date",
    # Investigations
    "bloods","bloods_checked","imaging","imaging_checked","ecg","microbiology","micro_checked",
    # Social / Discharge
    "social_situation","mobility","discharge_plan","discharge_barriers","expected_discharge",
    "tto_status","transport_booked","equipment_arranged","care_package_status","followup_booked",
    # Procedures
    "procedures","operation_date","post_op_day","case_classification",
    # Prep checklist
    "prep_bloods_done","prep_imaging_done","prep_nursing_done","prep_pharmacy_done",
    # Frailty / Risk
    "frailty_score","delirium_risk","falls_risk","pressure_injury_risk","sepsis_screening_date","aki_stage",
    # Alerts / Tags
    "alerts","tags",
    # Meta
    "status","created_at","last_updated","updated_by",
]

# ─────────────────────────────────────────────
# CSV MAPPING (ACS Theatre → MWIA)
# ─────────────────────────────────────────────
ACS_MAP = {
    "Transaction ID":      "patient_id",
    "Patient Sex":         "gender",
    "Age on Procedure Date":"age",
    "Surgery Date":        "operation_date",
    "Procedures":          "procedures",
    "Cardiothoracic Procedure":"admission_reason",
    "Primary Surgeon":     "consultant",
    "Location":            "hospital_site",
    "Theatre":             "ward",
    "Specialty":           "specialty",
    "Case Classification": "case_classification",
    "Patient Class":       "patient_class",
    "Primary An Type":     "dw_drugs",
    "Log Status":          "status",
    "NICOR":               "team",
}

def derive_priority(p: dict) -> str:
    """Derive priority from case classification or NEWS score"""
    news = int(p.get("news_score") or 0)
    if news >= 7: return "urgent"
    if news >= 5: return "amber"
    
    cc = (p.get("case_classification") or "").lower()
    if "p1a" in cc or "emergency" in cc: return "urgent"
    if "p1b" in cc or "urgent" in cc: return "amber"
    return "green"

def derive_sort_category(p: dict) -> str:
    """SORT: Sick / Out today / Rest / To come in"""
    # Sick = NEWS >=7 or escalation flag or critical
    news = int(p.get("news_score") or 0)
    if news >= 7 or p.get("escalation_flag") == "escalate" or p.get("priority") == "urgent":
        return "sick"
    
    # Out today = expected discharge today
    edd = p.get("expected_discharge","")
    if edd:
        try:
            parts = edd.split('/')
            if len(parts)==3:
                edd_date = date(int(parts[2]), int(parts[1]), int(parts[0]))
                if edd_date <= date.today():
                    return "out"
        except:
            pass
    
    # To come in = admission_date in future (not relevant for current patients, but structure for it)
    # Rest = everything else
    return "rest"

def derive_post_op(op_date_str: str) -> str:
    if not op_date_str: return ""
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]:
        try:
            op = datetime.strptime(op_date_str, fmt)
            d  = (datetime.now() - op).days
            return f"Day {d}" if d >= 0 else "Pre-op"
        except ValueError:
            pass
    return ""

def csv_row_to_patient(row: dict) -> dict:
    """Convert CSV row to patient dict with full schema support"""
    p: Dict[str, Any] = {f: "" for f in WARD_FIELDS}
    p["patient_id"]   = str(uuid.uuid4())
    p["created_at"]   = datetime.now().isoformat()
    p["last_updated"] = datetime.now().isoformat()
    p["status"]       = "Active"
    p["escalation_flag"] = "None"

    # Pass through ALL MWIA schema fields directly (MWIA CSV export)
    for field in WARD_FIELDS:
        val = str(row.get(field, "") or "").strip()
        if val and val.lower() not in ("nan", "none", ""):
            p[field] = val

    # Apply ACS Theatre column mapping (ACS CSV import)
    for csv_col, field in ACS_MAP.items():
        val = str(row.get(csv_col, "") or "").strip()
        if val and val.lower() not in ("nan", "none", ""):
            p[field] = val

    # Preserve patient_id from CSV if readable
    csv_pid = str(row.get("patient_id", "") or "").strip()
    if csv_pid and csv_pid.lower() not in ("nan","none",""):
        p["patient_id"] = csv_pid
    else:
        tx = str(row.get("Transaction ID", "")).strip()
        if tx: p["patient_id"] = f"TXN-{tx}"

    # Name
    name_val = str(row.get("name", "") or row.get("Name", "")).strip()
    if name_val and name_val.lower() not in ("nan", "none", ""):
        p["name"] = name_val
    else:
        tx = str(row.get("Transaction ID", "")).strip()
        p["name"] = f"Patient {tx}" if tx else f"Patient {p['patient_id']}"

    # Derive post-op day
    if not p.get("post_op_day"):
        p["post_op_day"] = derive_post_op(p.get("operation_date", ""))

    # Derive priority and SORT
    if not p.get("priority") or p["priority"] not in ("urgent","amber","green"):
        p["priority"] = derive_priority(p)
    p["sort_category"] = derive_sort_category(p)

    # Map gender
    g = p.get("gender","")
    if "[2]" in g or g.lower() == "m": p["gender"] = "Male"
    elif "[1]" in g or g.lower() == "f": p["gender"] = "Female"

    return p

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="MWIA v2", description="Modern Ward Round Intelligence Advisor", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
class LoginReq(BaseModel):
    username: str
    password: str

class PatientIn(BaseModel):
    """Full patient schema for create/update"""
    name: str
    # All other fields optional with defaults
    nhs_number: Optional[str] = ""
    dob: Optional[str] = ""
    age: Optional[str] = ""
    gender: Optional[str] = ""
    ward: Optional[str] = ""
    bay: Optional[str] = ""
    bed: Optional[str] = ""
    hospital_site: Optional[str] = ""
    consultant: Optional[str] = ""
    registrar: Optional[str] = ""
    team: Optional[str] = ""
    specialty: Optional[str] = ""
    admission_date: Optional[str] = ""
    admission_reason: Optional[str] = ""
    days_in: Optional[str] = ""
    patient_class: Optional[str] = ""
    priority: Optional[str] = "green"
    sort_category: Optional[str] = "rest"
    escalation_flag: Optional[str] = "None"
    dnacpr: Optional[str] = ""
    isolation: Optional[str] = ""
    vte_status: Optional[str] = ""
    # DAVID & WENDY
    dw_diet: Optional[str] = ""
    dw_activity: Optional[str] = ""
    dw_vitals: Optional[str] = ""
    dw_investigations: Optional[str] = ""
    dw_drains: Optional[str] = ""
    dw_wounds: Optional[str] = ""
    dw_examination: Optional[str] = ""
    dw_nursing: Optional[str] = ""
    dw_drugs: Optional[str] = ""
    dw_barriers: Optional[str] = ""
    # Obs
    obs_hr: Optional[str] = ""
    obs_hr_flag: Optional[str] = "ok"
    obs_bp: Optional[str] = ""
    obs_bp_flag: Optional[str] = "ok"
    obs_temp: Optional[str] = ""
    obs_temp_flag: Optional[str] = "ok"
    obs_o2: Optional[str] = ""
    obs_o2_flag: Optional[str] = "ok"
    obs_rr: Optional[str] = ""
    obs_rr_flag: Optional[str] = "ok"
    obs_consciousness: Optional[str] = ""
    obs_urine_output: Optional[str] = ""
    obs_pain_score: Optional[str] = ""
    news_score: Optional[str] = ""
    news_trend: Optional[str] = ""
    # Meds
    medications: Optional[str] = ""
    allergies: Optional[str] = ""
    antibiotics: Optional[str] = ""
    abx_indication: Optional[str] = ""
    abx_day: Optional[str] = ""
    abx_review: Optional[str] = ""
    abx_stop_date: Optional[str] = ""
    # Investigations
    bloods: Optional[str] = ""
    bloods_checked: Optional[str] = ""
    imaging: Optional[str] = ""
    imaging_checked: Optional[str] = ""
    ecg: Optional[str] = ""
    microbiology: Optional[str] = ""
    micro_checked: Optional[str] = ""
    # Social/DC
    social_situation: Optional[str] = ""
    mobility: Optional[str] = ""
    discharge_plan: Optional[str] = ""
    discharge_barriers: Optional[str] = ""
    expected_discharge: Optional[str] = ""
    tto_status: Optional[str] = ""
    transport_booked: Optional[str] = ""
    equipment_arranged: Optional[str] = ""
    care_package_status: Optional[str] = ""
    followup_booked: Optional[str] = ""
    # Procedures
    procedures: Optional[str] = ""
    operation_date: Optional[str] = ""
    post_op_day: Optional[str] = ""
    case_classification: Optional[str] = ""
    # Prep
    prep_bloods_done: Optional[str] = ""
    prep_imaging_done: Optional[str] = ""
    prep_nursing_done: Optional[str] = ""
    prep_pharmacy_done: Optional[str] = ""
    # Risk
    frailty_score: Optional[str] = ""
    delirium_risk: Optional[str] = ""
    falls_risk: Optional[str] = ""
    pressure_injury_risk: Optional[str] = ""
    sepsis_screening_date: Optional[str] = ""
    aki_stage: Optional[str] = ""
    # Meta
    alerts: Optional[str] = ""
    tags: Optional[str] = ""
    status: Optional[str] = "Active"

class NoteIn(BaseModel):
    patient_id: str
    note_type: str
    note_text: str
    author: Optional[str] = ""

class TaskIn(BaseModel):
    patient_id: str
    task_text: str
    category: str  # bloods/imaging/referral/meds/discharge/nursing/therapy/other
    owner: str     # doctor/nurse/pharmacy/therapy/other
    due_date: Optional[str] = ""
    due_time: Optional[str] = ""
    priority: Optional[str] = "normal"  # urgent/high/normal/low

class TaskUpdate(BaseModel):
    status: Optional[str] = None  # open/in-progress/done/blocked
    blocker_reason: Optional[str] = None
    completed_by: Optional[str] = None

class SafetyBundleIn(BaseModel):
    patient_id: str
    # VTE
    vte_indicated: Optional[str] = ""
    vte_type: Optional[str] = ""
    vte_prescribed: Optional[str] = ""
    # Antibiotics
    abx_indication: Optional[str] = ""
    abx_day: Optional[str] = ""
    abx_review_date: Optional[str] = ""
    abx_stop_date: Optional[str] = ""
    # Lines/Catheters
    lines_list: Optional[str] = ""
    lines_review_plan: Optional[str] = ""
    # Escalation
    ceiling_of_care: Optional[str] = ""
    dnacpr_status: Optional[str] = ""
    dnacpr_documented: Optional[str] = ""
    escalation_plan_clear: Optional[str] = ""

class MDTInputIn(BaseModel):
    patient_id: str
    role: str  # nursing/pharmacy/therapy
    input_text: str

class AIReq(BaseModel):
    message: str
    patient_id: Optional[str] = ""
    mode: Optional[str] = "bedside"  # board/bedside/post-round
    history: Optional[List[Dict]] = []

class BoardRoundStart(BaseModel):
    ward: str
    team: str
    attendees: List[str]  # ["Dr Shah", "Nurse Jane", "Pharmacist John"]

# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────
@app.post("/api/login")
async def login(req: LoginReq, response: Response):
    if not verify_login(req.username, req.password):
        raise HTTPException(401, "Invalid credentials")
    tok = create_session(req.username)
    response.set_cookie("session_token", tok, httponly=True, max_age=28800, samesite="lax")
    return {"status": "ok", "user": req.username}

@app.post("/api/logout")
async def logout(response: Response, session_token: Optional[str] = Cookie(default=None)):
    active_sessions.pop(session_token or "", None)
    response.delete_cookie("session_token")
    return {"status": "logged_out"}

@app.get("/api/me")
async def me(session_token: Optional[str] = Cookie(default=None)):
    u = get_user(session_token)
    if not u: raise HTTPException(401, "Not authenticated")
    return {"user": u, "ai_ready": bool(openai_client)}

# ─────────────────────────────────────────────
# CSV IMPORT (optimized with validation)
# ─────────────────────────────────────────────
@app.post("/api/import-csv")
async def import_csv(file: UploadFile = File(...),
                     session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    content = await file.read()
    
    # Validate size
    if len(content) > 10_000_000:  # 10MB limit
        raise HTTPException(400, "CSV too large (max 10MB)")
    
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"CSV parse error: {e}")

    if df.empty:
        raise HTTPException(400, "CSV is empty")

    imported = 0
    errors = []
    for idx, row in df.iterrows():
        try:
            p = csv_row_to_patient(row.to_dict())
            pid = p["patient_id"]
            patients[pid]       = p
            round_notes[pid]    = []
            safety_checks[pid]  = {}
            mdt_inputs[pid]     = {"nursing":"", "pharmacy":"", "therapy":""}
            imported += 1
        except Exception as e:
            errors.append(f"Row {idx+2}: {str(e)}")
            if len(errors) >= 10:  # Stop after 10 errors
                break

    return {
        "status": "imported",
        "count": imported,
        "total_rows": len(df),
        "errors": errors[:10] if errors else []
    }

# ─────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────
@app.get("/api/export-csv")
async def export_csv(session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    if not patients:
        raise HTTPException(404, "No patients to export")

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=WARD_FIELDS, extrasaction="ignore")
    w.writeheader()
    for p in patients.values():
        w.writerow(p)

    buf.seek(0)
    fname = f"mwia_ward_round_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return StreamingResponse(io.StringIO(buf.getvalue()), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={fname}"})

@app.get("/api/csv-template")
async def csv_template():
    """Download blank CSV template"""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=WARD_FIELDS, extrasaction="ignore")
    w.writeheader()
    buf.seek(0)
    return StreamingResponse(io.StringIO(buf.getvalue()), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=mwia_template.csv"})

# ─────────────────────────────────────────────
# PATIENT CRUD
# ─────────────────────────────────────────────
@app.get("/api/patients")
async def get_patients(ward: Optional[str] = None, priority: Optional[str] = None,
                       sort_cat: Optional[str] = None, search: Optional[str] = None,
                       session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    pts = list(patients.values())
    
    if ward and ward != "all":
        pts = [p for p in pts if p.get("ward","").lower() == ward.lower()]
    if priority and priority != "all":
        pts = [p for p in pts if p.get("priority","") == priority]
    if sort_cat and sort_cat != "all":
        pts = [p for p in pts if p.get("sort_category","") == sort_cat]
    if search:
        s = search.lower()
        pts = [p for p in pts if any(s in str(p.get(f,"")).lower()
               for f in ["name","admission_reason","nhs_number","bed","consultant","ward","specialty"])]
    
    # Sort by SORT category priority: sick > out > rest
    sort_order = {"sick":0, "out":1, "rest":2, "to-come-in":3}
    pts.sort(key=lambda p: (sort_order.get(p.get("sort_category","rest"), 2), p.get("name","")))
    
    return {"patients": pts, "count": len(pts)}

@app.get("/api/patients/{pid}")
async def get_patient(pid: str, session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    p = patients.get(pid)
    if not p: raise HTTPException(404, "Patient not found")
    return {
        "patient": p,
        "notes": round_notes.get(pid, []),
        "safety": safety_checks.get(pid, {}),
        "mdt": mdt_inputs.get(pid, {}),
        "tasks": [t for t in tasks.values() if t.get("patient_id") == pid]
    }

@app.post("/api/patients")
async def create_patient(p: PatientIn, session_token: Optional[str] = Cookie(default=None)):
    u = require_user(session_token)
    pid = str(uuid.uuid4())
    rec = p.dict()
    rec["patient_id"]   = pid
    rec["created_at"]   = datetime.now().isoformat()
    rec["last_updated"] = datetime.now().isoformat()
    rec["updated_by"]   = u
    
    if not rec.get("post_op_day") and rec.get("operation_date"):
        rec["post_op_day"] = derive_post_op(rec["operation_date"])
    if not rec.get("priority") or rec["priority"] not in ("urgent","amber","green"):
        rec["priority"] = derive_priority(rec)
    rec["sort_category"] = derive_sort_category(rec)
    
    patients[pid] = rec
    round_notes[pid] = []
    safety_checks[pid] = {}
    mdt_inputs[pid] = {"nursing":"", "pharmacy":"", "therapy":""}
    return {"status": "created", "patient_id": pid}

@app.put("/api/patients/{pid}")
async def update_patient(pid: str, p: PatientIn,
                         session_token: Optional[str] = Cookie(default=None)):
    u = require_user(session_token)
    if pid not in patients: raise HTTPException(404, "Patient not found")
    
    rec = p.dict()
    rec["patient_id"]   = pid
    rec["created_at"]   = patients[pid].get("created_at", datetime.now().isoformat())
    rec["last_updated"] = datetime.now().isoformat()
    rec["updated_by"]   = u
    
    if not rec.get("post_op_day") and rec.get("operation_date"):
        rec["post_op_day"] = derive_post_op(rec["operation_date"])
    if not rec.get("priority") or rec["priority"] not in ("urgent","amber","green"):
        rec["priority"] = derive_priority(rec)
    rec["sort_category"] = derive_sort_category(rec)
    
    patients[pid] = rec
    return {"status": "updated"}

@app.delete("/api/patients/{pid}")
async def delete_patient(pid: str, session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    if pid not in patients: raise HTTPException(404, "Not found")
    del patients[pid]
    round_notes.pop(pid, None)
    safety_checks.pop(pid, None)
    mdt_inputs.pop(pid, None)
    # Delete associated tasks
    for tid in list(tasks.keys()):
        if tasks[tid].get("patient_id") == pid:
            del tasks[tid]
    return {"status": "deleted"}

# ─────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────
@app.post("/api/notes")
async def add_note(n: NoteIn, session_token: Optional[str] = Cookie(default=None)):
    u = require_user(session_token)
    pid = n.patient_id
    if pid not in patients: raise HTTPException(404, "Patient not found")
    
    note = {
        "note_id": str(uuid.uuid4()),
        "patient_id": pid,
        "note_type": n.note_type,
        "note_text": n.note_text,
        "author": n.author or u,
        "timestamp": datetime.now().isoformat()
    }
    round_notes.setdefault(pid, []).append(note)
    return {"status": "saved", "note_id": note["note_id"]}

@app.get("/api/notes/{pid}")
async def get_notes(pid: str, session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    return {"notes": round_notes.get(pid, [])}

# ─────────────────────────────────────────────
# TASKS SYSTEM
# ─────────────────────────────────────────────
@app.post("/api/tasks")
async def create_task(t: TaskIn, session_token: Optional[str] = Cookie(default=None)):
    u = require_user(session_token)
    if t.patient_id and t.patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    
    tid = str(uuid.uuid4())
    task = {
        "task_id": tid,
        "patient_id": t.patient_id,
        "task_text": t.task_text,
        "category": t.category,
        "owner": t.owner,
        "due_date": t.due_date,
        "due_time": t.due_time,
        "priority": t.priority,
        "status": "open",
        "blocker_reason": "",
        "completed_by": "",
        "created_at": datetime.now().isoformat(),
        "created_by": u
    }
    tasks[tid] = task
    return {"status": "created", "task_id": tid}

@app.get("/api/tasks")
async def get_tasks(patient_id: Optional[str] = None, owner: Optional[str] = None,
                    status: Optional[str] = None, category: Optional[str] = None,
                    session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    
    result = list(tasks.values())
    if patient_id:
        result = [t for t in result if t.get("patient_id") == patient_id]
    if owner and owner != "all":
        result = [t for t in result if t.get("owner") == owner]
    if status and status != "all":
        result = [t for t in result if t.get("status") == status]
    if category and category != "all":
        result = [t for t in result if t.get("category") == category]
    
    # Sort by priority then due date
    priority_order = {"urgent":0, "high":1, "normal":2, "low":3}
    result.sort(key=lambda t: (
        priority_order.get(t.get("priority","normal"), 2),
        t.get("due_date","") or "9999-12-31"
    ))
    
    return {"tasks": result, "count": len(result)}

@app.put("/api/tasks/{tid}")
async def update_task(tid: str, upd: TaskUpdate,
                      session_token: Optional[str] = Cookie(default=None)):
    u = require_user(session_token)
    if tid not in tasks: raise HTTPException(404, "Task not found")
    
    if upd.status is not None:
        tasks[tid]["status"] = upd.status
        if upd.status == "done":
            tasks[tid]["completed_by"] = u
            tasks[tid]["completed_at"] = datetime.now().isoformat()
    
    if upd.blocker_reason is not None:
        tasks[tid]["blocker_reason"] = upd.blocker_reason
    
    return {"status": "updated"}

@app.delete("/api/tasks/{tid}")
async def delete_task(tid: str, session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    if tid not in tasks: raise HTTPException(404, "Task not found")
    del tasks[tid]
    return {"status": "deleted"}

# ─────────────────────────────────────────────
# SAFETY BUNDLES
# ─────────────────────────────────────────────
@app.post("/api/safety-bundle")
async def save_safety_bundle(sb: SafetyBundleIn,
                              session_token: Optional[str] = Cookie(default=None)):
    u = require_user(session_token)
    pid = sb.patient_id
    if pid not in patients: raise HTTPException(404, "Patient not found")
    
    bundle = sb.dict()
    bundle["last_updated"] = datetime.now().isoformat()
    bundle["updated_by"] = u
    safety_checks[pid] = bundle
    return {"status": "saved"}

@app.get("/api/safety-bundle/{pid}")
async def get_safety_bundle(pid: str, session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    return {"bundle": safety_checks.get(pid, {})}

# ─────────────────────────────────────────────
# MDT INPUTS
# ─────────────────────────────────────────────
@app.post("/api/mdt-input")
async def save_mdt_input(mdt: MDTInputIn,
                         session_token: Optional[str] = Cookie(default=None)):
    u = require_user(session_token)
    pid = mdt.patient_id
    if pid not in patients: raise HTTPException(404, "Patient not found")
    
    if pid not in mdt_inputs:
        mdt_inputs[pid] = {"nursing":"", "pharmacy":"", "therapy":""}
    
    mdt_inputs[pid][mdt.role] = mdt.input_text
    return {"status": "saved"}

@app.get("/api/mdt-input/{pid}")
async def get_mdt_input(pid: str, session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    return {"mdt": mdt_inputs.get(pid, {})}

# ─────────────────────────────────────────────
# BOARD ROUND
# ─────────────────────────────────────────────
@app.post("/api/board-round/start")
async def start_board_round(br: BoardRoundStart,
                            session_token: Optional[str] = Cookie(default=None)):
    u = require_user(session_token)
    global board_round_session
    board_round_session = {
        "date": date.today().isoformat(),
        "ward": br.ward,
        "team": br.team,
        "attendees": br.attendees,
        "started_at": datetime.now().isoformat(),
        "started_by": u
    }
    return {"status": "started", "session": board_round_session}

@app.get("/api/board-round/session")
async def get_board_round_session(session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    return {"session": board_round_session}

@app.post("/api/board-round/end")
async def end_board_round(session_token: Optional[str] = Cookie(default=None)):
    u = require_user(session_token)
    global board_round_session
    if board_round_session:
        board_round_session["ended_at"] = datetime.now().isoformat()
        board_round_session["ended_by"] = u
    session = board_round_session
    board_round_session = None
    return {"status": "ended", "session": session}

# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
@app.get("/api/stats")
async def stats(session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    pts = list(patients.values())
    
    # SORT counts
    sick = sum(1 for p in pts if p.get("sort_category") == "sick")
    out  = sum(1 for p in pts if p.get("sort_category") == "out")
    rest = sum(1 for p in pts if p.get("sort_category") == "rest")
    
    # Tasks
    open_tasks = sum(1 for t in tasks.values() if t.get("status") == "open")
    urgent_tasks = sum(1 for t in tasks.values() if t.get("priority") == "urgent" and t.get("status") != "done")
    
    return {
        "total":    len(pts),
        "sick":     sick,
        "out":      out,
        "rest":     rest,
        "urgent":   sum(1 for p in pts if p.get("priority") == "urgent"),
        "amber":    sum(1 for p in pts if p.get("priority") == "amber"),
        "stable":   sum(1 for p in pts if p.get("priority") == "green"),
        "wards":    list(set(p.get("ward","") for p in pts if p.get("ward"))),
        "tasks_open": open_tasks,
        "tasks_urgent": urgent_tasks,
        "timestamp": datetime.now().isoformat(),
    }

# ─────────────────────────────────────────────
# VOICE — STT (Whisper)
# ─────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...),
                     session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    if not openai_client:
        raise HTTPException(503, "OpenAI API key not configured")

    audio_bytes = await audio.read()
    suffix = ".webm"
    if audio.filename:
        ext = Path(audio.filename).suffix
        if ext: suffix = ext

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = openai_client.audio.transcriptions.create(
                model=STT_MODEL, file=f, language="en"
            )
        text = result.text.strip()
    finally:
        os.unlink(tmp_path)

    return {"text": text}

# ─────────────────────────────────────────────
# VOICE — TTS (OpenAI)
# ─────────────────────────────────────────────
@app.post("/api/speak")
async def speak(request: Request, session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    if not openai_client:
        raise HTTPException(503, "OpenAI API key not configured")

    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(400, "No text provided")

    # Strip markdown and limit length
    clean = re.sub(r'[*_#`\[\]()]', '', text).replace('\n', ' ').strip()
    clean = clean[:4000]

    voice = body.get("voice", TTS_VOICE)

    response = openai_client.audio.speech.create(
        model=TTS_MODEL, voice=voice, input=clean,
        response_format="mp3", speed=0.92
    )

    return StreamingResponse(io.BytesIO(response.content),
                             media_type="audio/mpeg",
                             headers={"Content-Disposition": "inline; filename=speech.mp3"})

# ─────────────────────────────────────────────
# AI — GPT-4o clinical reasoning
# ─────────────────────────────────────────────
@app.post("/api/ai")
async def ai_query(req: AIReq, session_token: Optional[str] = Cookie(default=None)):
    require_user(session_token)
    if not openai_client:
        raise HTTPException(503, "OpenAI API key not configured")

    # Build patient context
    pt_context = "No patient currently selected."
    if req.patient_id and req.patient_id in patients:
        p = patients[req.patient_id]
        safety = safety_checks.get(req.patient_id, {})
        mdt = mdt_inputs.get(req.patient_id, {})
        pt_tasks = [t for t in tasks.values() if t.get("patient_id") == req.patient_id and t.get("status") != "done"]
        
        pt_context = f"""
CURRENT PATIENT: {p.get('name','')}
Ward: {p.get('ward','')} | Bay: {p.get('bay','')} | Bed: {p.get('bed','')}
Age: {p.get('age','')} | Gender: {p.get('gender','')} | NHS: {p.get('nhs_number','')}
Consultant: {p.get('consultant','')} | Team: {p.get('team','')}
Admission: {p.get('admission_date','')} — {p.get('admission_reason','')} (Day {p.get('days_in','?')})
SORT Category: {p.get('sort_category','').upper()} | Priority: {p.get('priority','').upper()}
Post-op: {p.get('post_op_day','')} | Operation: {p.get('procedures','')}

DAVID & WENDY FRAMEWORK:
D – Diet:           {p.get('dw_diet','')}
A – Activity:       {p.get('dw_activity','')}
V – Vitals:         {p.get('dw_vitals','')}
I – Investigations: {p.get('dw_investigations','')}
D – Drains/Lines:   {p.get('dw_drains','')}
W – Wound:          {p.get('dw_wounds','')}
E – Examination:    {p.get('dw_examination','')}
N – Nursing:        {p.get('dw_nursing','')}
D – Drugs:          {p.get('dw_drugs','')}
Y – Barriers:       {p.get('dw_barriers','')}

OBSERVATIONS: HR {p.get('obs_hr','')} | BP {p.get('obs_bp','')} | Temp {p.get('obs_temp','')} | O2 {p.get('obs_o2','')} | RR {p.get('obs_rr','')} | NEWS {p.get('news_score','')}
MEDICATIONS: {p.get('medications','')}
ALLERGIES: {p.get('allergies','')}
ANTIBIOTICS: {p.get('antibiotics','')} — Indication: {p.get('abx_indication','')} | Day {p.get('abx_day','')} | Review: {p.get('abx_review','')}
VTE: {p.get('vte_status','')}
BLOODS: {p.get('bloods','')}
IMAGING: {p.get('imaging','')}
MICROBIOLOGY: {p.get('microbiology','')}
SOCIAL: {p.get('social_situation','')}
DISCHARGE PLAN: {p.get('discharge_plan','')} — EDD: {p.get('expected_discharge','')} | Barriers: {p.get('discharge_barriers','')}
ALERTS: {p.get('alerts','')}
DNACPR: {p.get('dnacpr','')} | ISOLATION: {p.get('isolation','')}

SAFETY BUNDLE:
VTE Prophylaxis: {safety.get('vte_indicated','')} | Type: {safety.get('vte_type','')} | Prescribed: {safety.get('vte_prescribed','')}
Lines/Catheters: {safety.get('lines_list','')} | Review Plan: {safety.get('lines_review_plan','')}
Ceiling of Care: {safety.get('ceiling_of_care','')} | DNACPR: {safety.get('dnacpr_status','')}

MDT INPUTS:
Nursing: {mdt.get('nursing','')}
Pharmacy: {mdt.get('pharmacy','')}
Therapy: {mdt.get('therapy','')}

OUTSTANDING TASKS: {len(pt_tasks)} tasks | {', '.join(t.get('task_text','')[:40] for t in pt_tasks[:3])}
""".strip()

    # Mode-specific system prompts
    now = datetime.now().strftime("%A %d %B %Y, %H:%M")
    
    if req.mode == "board":
        mode_context = """
MODE: Board Round (Pre-round huddle)
- Provide concise SORT-based summary (Sick/Out/Rest)
- Highlight red flags, acuity, discharge blockers
- Suggest priorities for bedside review
- Keep to 2-3 sentences maximum
"""
    elif req.mode == "post-round":
        mode_context = """
MODE: Post-Round Jobs Generation
- Generate structured task list from the conversation
- Specify owner, category, priority for each task
- Flag urgent actions clearly
- Group by role (doctor/nurse/pharmacy/therapy)
"""
    else:  # bedside
        mode_context = """
MODE: Bedside Review
- Spoken-word optimised: 3-5 sentences maximum
- Lead with most important finding first
- Use real numbers from patient data
- Flag urgency clearly
- End with single clear recommended action
- For note generation: produce full structured DAVID & WENDY note
"""

    system = f"""You are MWIA — Modern Ward Round Intelligence Advisor.
Date/time: {now}

You are the voice-first AI clinical intelligence system for surgical/medical ward rounds at King's College Hospital NHS Foundation Trust. You speak DIRECTLY TO THE CONSULTANT during their ward round.

{mode_context}

{pt_context}

YOUR ROLE:
- Summarise patient status concisely
- Flag clinical alerts and red flags immediately
- Review medications, VTE prophylaxis, antibiotic stewardship
- Assess discharge readiness against DAVID & WENDY criteria
- Generate structured ward round notes when requested
- Help create task lists with clear ownership

YOUR BEHAVIOUR:
- If data is missing, say "data not available" — NEVER invent
- Use real clinical numbers from the patient data
- For bedside: spoken English, conversational, no bullet points
- For board round: ultra-concise SORT summary
- For post-round: structured task list with ownership
- This is a development prototype with synthetic data — not for real clinical use

If asked to generate a note, produce BOTH:
1. Clinical note (for clinicians, structured, DAVID & WENDY format)
2. Patient summary (plain English, what's happening today)"""

    messages = [{"role": "system", "content": system}]
    for h in (req.history or [])[-8:]:
        if h.get("role") in ("user","assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    completion = openai_client.chat.completions.create(
        model=AI_MODEL, messages=messages, max_tokens=1500, temperature=0.3
    )
    reply = completion.choices[0].message.content.strip()
    return {"reply": reply}

# ─────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve():
    p = Path("mwia_frontend_v2.html")
    if p.exists():
        return HTMLResponse(content=p.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>MWIA v2 — Frontend not found. Place mwia_frontend_v2.html alongside this file.</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
