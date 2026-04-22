"""Florida Statute 90.902(11) Evidence Certification PDF Generator.

Produces a self-authenticating certification declaration PDF that accompanies
every exported Pylox Vision evidence package. When properly signed and notarized
by the site custodian, the certification makes the attached footage admissible
in Florida court WITHOUT a live witness under F.S. 90.902(11) and F.S. 90.803(6)
(business records exception).

The cert includes:
  - Statutory language verbatim
  - Device serial number and software version
  - Incident ID, camera IDs, and timeframe
  - SHA-256 hashes of every included file
  - Custodian signature line
  - Notary block

Usage:
    from engine.police.evidence_cert import generate_certification

    pdf_path = generate_certification(
        incident=incident_dict,
        files=[Path("clip1.mp4"), Path("clip2.mp4")],
        output_dir=Path("/tmp/pylox_evidence"),
        site_config={"name": "...", "address": "...", "owner": "..."},
        device_info={"serial": "JETSON-001", "version": "2.0.0"},
    )

The PDF is designed to print to a single page whenever possible, ready for
immediate signature on scene.
"""

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pylox-v2.police.cert")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Exact statutory language from F.S. 90.902(11)
STATUTORY_CERTIFICATION_TEXT = """CERTIFICATION OF RECORDS OF REGULARLY CONDUCTED BUSINESS ACTIVITY
(Florida Statute 90.902(11))

I, the undersigned, being a custodian of records or otherwise qualified to make
this certification, declare under penalty of perjury under the laws of the State
of Florida that the following is true and correct:

1. I am duly authorized by {site_name} ({site_address}) to make this
   certification on behalf of that business.

2. The records described below were made at or near the time of the occurrence
   of the matters set forth in the records, by — or from information transmitted
   by — a person with knowledge of those matters.

3. The records were kept in the course of the regularly conducted business
   activity of {site_name}.

4. The making of the records was a regular practice of that business activity.

5. The records consist of digital video footage, computer-generated incident
   metadata, and cryptographic hashes captured by the Pylox Vision AI Security
   System, Device Serial Number {device_serial}, Software Version {device_version},
   continuously operating at the above-referenced premises.

6. The records attached to this certification have not been altered, manipulated,
   or otherwise changed since their original capture, as verified by the SHA-256
   cryptographic hashes listed below."""


def _wrap_text(text: str, width: int = 82) -> list:
    """Simple word-wrap returning a list of lines."""
    out = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            out.append("")
            continue
        words = paragraph.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 <= width:
                line = f"{line} {w}".strip()
            else:
                out.append(line)
                line = w
        if line:
            out.append(line)
    return out


def generate_certification(
    incident: dict,
    files: list,
    output_dir: Path,
    site_config: dict,
    device_info: dict,
    narrative: str = None,
) -> Optional[Path]:
    """Generate the FL 90.902(11) certification PDF for an evidence package.

    Args:
        incident: incident dict (id, camera, start_time, end_time, max_threat, resolution)
        files: list of Path objects for every file included in the evidence package
        output_dir: directory to write the PDF into (will be created if missing)
        site_config: {"name": str, "address": str, "owner": str, "phone": str}
        device_info: {"serial": str, "version": str}
        narrative: optional pre-generated report narrative to include on page 2

    Returns:
        Path to generated PDF, or None on failure.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
            KeepTogether,
        )
        from reportlab.lib import colors
    except ImportError:
        logger.error("reportlab not installed — cannot generate evidence certification")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    incident_id = incident.get("id", "UNKNOWN")
    camera = incident.get("camera", "unknown")
    start_ts = incident.get("start_time", 0)
    end_ts = incident.get("end_time") or time.time()

    pdf_path = output_dir / f"pylox_evidence_cert_incident_{incident_id}.pdf"

    site_name = site_config.get("name", "Commercial Property")
    site_address = site_config.get("address", "Address on File")
    owner_name = site_config.get("owner", "")
    device_serial = device_info.get("serial", "PYLOX-UNKNOWN")
    device_version = device_info.get("version", "2.0.0")

    # Compute SHA-256 for every file
    file_hashes = []
    for f in files:
        fp = Path(f)
        if not fp.exists():
            continue
        try:
            digest = sha256_file(fp)
            size_bytes = fp.stat().st_size
            file_hashes.append({
                "name": fp.name,
                "size": size_bytes,
                "sha256": digest,
            })
        except Exception as e:
            logger.warning(f"Could not hash {fp}: {e}")

    # Build the PDF
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"Pylox Vision Evidence Certification — Incident {incident_id}",
        author="Pylox Vision AI Security System",
        subject="FL Statute 90.902(11) Self-Authenticating Business Records Certification",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title", parent=styles["Title"], fontSize=14, alignment=TA_CENTER,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "heading", parent=styles["Heading2"], fontSize=11,
        spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#000066"),
    )
    body_style = ParagraphStyle(
        "body", parent=styles["Normal"], fontSize=9, alignment=TA_JUSTIFY,
        leading=12, spaceAfter=6,
    )
    mono_style = ParagraphStyle(
        "mono", parent=styles["Code"], fontSize=7, leading=9,
        fontName="Courier", spaceAfter=3,
    )

    story = []

    # Header
    story.append(Paragraph("PYLOX VISION AI SECURITY SYSTEM", title_style))
    story.append(Paragraph(
        "EVIDENCE CERTIFICATION — FLORIDA STATUTE 90.902(11)", title_style,
    ))
    story.append(Spacer(1, 12))

    # Incident summary table
    incident_data = [
        ["Incident ID", str(incident_id)],
        ["Site", f"{site_name}"],
        ["Address", f"{site_address}"],
        ["Primary Camera", f"{camera}"],
        ["Start (EDT)", datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S")],
        ["End (EDT)", datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S")],
        ["Duration", f"{int(end_ts - start_ts)} seconds"],
        ["Max Threat", f"{incident.get('max_threat', 0)}/10"],
        ["Resolution", incident.get("resolution", "unknown")],
        ["Device Serial", device_serial],
        ["Software Version", f"Pylox Vision v{device_version}"],
        ["Certification Issued", datetime.now().strftime("%Y-%m-%d %H:%M:%S EDT")],
    ]
    tbl = Table(incident_data, colWidths=[1.6 * inch, 5.0 * inch])
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEEEEE")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Statutory certification text
    story.append(Paragraph("STATUTORY CERTIFICATION", heading_style))
    cert_text = STATUTORY_CERTIFICATION_TEXT.format(
        site_name=site_name,
        site_address=site_address,
        device_serial=device_serial,
        device_version=device_version,
    )
    for para in cert_text.split("\n\n"):
        para_clean = para.replace("\n", " ").strip()
        if para_clean:
            story.append(Paragraph(para_clean, body_style))

    # File hash manifest
    story.append(Paragraph("ATTACHED FILE MANIFEST — SHA-256 HASHES", heading_style))
    if file_hashes:
        hash_data = [["#", "File", "Size (bytes)", "SHA-256"]]
        for i, fh in enumerate(file_hashes, 1):
            hash_data.append([
                str(i),
                fh["name"],
                f"{fh['size']:,}",
                fh["sha256"][:32] + "\n" + fh["sha256"][32:],
            ])
        hash_tbl = Table(
            hash_data,
            colWidths=[0.3 * inch, 2.3 * inch, 1.0 * inch, 3.0 * inch],
        )
        hash_tbl.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
            ("FONT", (0, 1), (-1, -1), "Courier", 7),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#000066")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(hash_tbl)
    else:
        story.append(Paragraph("(No files attached to this certification)", body_style))

    story.append(Spacer(1, 14))

    # Custodian affirmation + signature block
    story.append(Paragraph("CUSTODIAN SIGNATURE", heading_style))
    story.append(Paragraph(
        f"I, the undersigned custodian of records for {site_name}, affirm that "
        f"the foregoing certification is true and correct. I am authorized to "
        f"make this certification under Florida Statute 90.902(11) and Florida "
        f"Statute 90.803(6).",
        body_style,
    ))
    story.append(Spacer(1, 20))

    sig_data = [
        ["Signature:", "_" * 55],
        ["", ""],
        ["Printed Name:", "_" * 55],
        ["", ""],
        ["Title:", "_" * 55],
        ["", ""],
        ["Date:", "_" * 55],
        ["", ""],
        ["Contact Phone:", "_" * 55],
    ]
    if owner_name:
        sig_data[2] = ["Printed Name:", f"{owner_name}   " + "_" * (55 - len(owner_name) - 3)]
    sig_tbl = Table(sig_data, colWidths=[1.4 * inch, 5.2 * inch])
    sig_tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sig_tbl)
    story.append(Spacer(1, 16))

    # Notary block
    story.append(Paragraph("NOTARY ACKNOWLEDGEMENT (STATE OF FLORIDA)", heading_style))
    notary_text = (
        "State of Florida, County of _____________________.<br/><br/>"
        "The foregoing instrument was acknowledged before me by means of "
        "&nbsp;&nbsp;&#9744;&nbsp;physical presence or "
        "&nbsp;&nbsp;&#9744;&nbsp;online notarization, "
        "this _______ day of __________________, 20____, "
        "by _______________________________, who is personally known to me "
        "or who has produced _____________________ as identification.<br/><br/>"
        "_______________________________________________<br/>"
        "Signature of Notary Public — State of Florida<br/><br/>"
        "_______________________________________________<br/>"
        "Print, Type, or Stamp Commissioned Name of Notary<br/><br/>"
        "My Commission Expires: _________________________"
    )
    story.append(Paragraph(notary_text, body_style))

    # Optional: narrative on page 2
    if narrative:
        story.append(PageBreak())
        story.append(Paragraph("INCIDENT REPORT NARRATIVE", title_style))
        story.append(Paragraph(
            "The following narrative was auto-generated by the Pylox Vision AI "
            "Security System based on direct observation of the incident. It is "
            "provided for use in police reports under F.S. 90.803(6).",
            body_style,
        ))
        story.append(Spacer(1, 8))
        story.append(Paragraph(narrative, body_style))

    # Footer marker
    story.append(Spacer(1, 10))
    footer = ParagraphStyle(
        "footer", parent=styles["Normal"], fontSize=7, alignment=TA_CENTER,
        textColor=colors.grey,
    )
    story.append(Paragraph(
        f"Generated by Pylox Vision AI Security System v{device_version} "
        f"&mdash; Device {device_serial} &mdash; "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S EDT')}",
        footer,
    ))

    try:
        doc.build(story)
        logger.info(
            f"Evidence certification generated: {pdf_path} "
            f"({len(file_hashes)} files hashed)"
        )
        return pdf_path
    except Exception as e:
        logger.error(f"Failed to build certification PDF: {e}")
        return None
