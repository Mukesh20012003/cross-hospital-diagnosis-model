# server/reports.py

import os
import csv
import json
from datetime import datetime
from typing import List, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.platypus import PageBreak


def generate_compliance_report(
    rounds_data: List[dict],
    hospitals_data: List[dict],
    audit_logs: List[dict],
    privacy_params: dict,
    output_path: str = "reports/compliance_report.pdf"
) -> str:
    """
    Generate a professional PDF compliance report for regulators.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1a237e')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#283593')
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        textColor=colors.HexColor('#333333')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leading=14
    )
    
    elements = []
    
    # ---- TITLE PAGE ----
    elements.append(Spacer(1, 100))
    elements.append(Paragraph("Cross-Hospital Diagnosis Model", title_style))
    elements.append(Paragraph("Federated Learning Compliance Report", heading_style))
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", 
        normal_style
    ))
    elements.append(Paragraph(
        "Classification: CONFIDENTIAL - For Regulatory Review Only", 
        normal_style
    ))
    elements.append(Spacer(1, 50))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    summary_text = (
        "This report documents the federated learning training process for the "
        "Cross-Hospital Diagnosis Model. The system enables multiple hospitals to "
        "collaboratively train a diagnostic AI model WITHOUT sharing raw patient data. "
        "Only model weights (numerical parameters) are exchanged between hospital nodes "
        "and the central aggregation server. This approach ensures full compliance with "
        "data privacy regulations including HIPAA and GDPR."
    )
    elements.append(Paragraph(summary_text, normal_style))
    
    elements.append(PageBreak())
    
    # ---- PRIVACY SECTION ----
    elements.append(Paragraph("1. Privacy & Security Measures", heading_style))
    
    elements.append(Paragraph("1.1 Data Privacy Guarantee", subheading_style))
    privacy_text = (
        "• No raw patient data (images, records, demographics) is transmitted between institutions.<br/>"
        "• Only model weight updates (numerical tensors) are shared.<br/>"
        "• Each hospital trains on its own local data partition.<br/>"
        "• The aggregation server never has access to patient-level data.<br/>"
        "• All communications are logged for audit purposes."
    )
    elements.append(Paragraph(privacy_text, normal_style))
    
    elements.append(Paragraph("1.2 Differential Privacy Parameters", subheading_style))
    if privacy_params:
        privacy_table_data = [["Parameter", "Value"]]
        for key, value in privacy_params.items():
            if key != "description":
                privacy_table_data.append([str(key), str(value)])
        
        privacy_table = Table(privacy_table_data, colWidths=[200, 280])
        privacy_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        elements.append(privacy_table)
    
    elements.append(Spacer(1, 20))
    
    # ---- TRAINING ROUNDS SECTION ----
    elements.append(Paragraph("2. Training Rounds Summary", heading_style))
    
    if rounds_data:
        rounds_table_data = [["Round", "Status", "Participants", "Accuracy", "Loss", "Date"]]
        for r in rounds_data:
            rounds_table_data.append([
                str(r.get('round_number', '')),
                str(r.get('status', '')),
                str(r.get('num_participants', '')),
                f"{r['global_accuracy']*100:.2f}%" if r.get('global_accuracy') else 'N/A',
                f"{r['global_loss']:.4f}" if r.get('global_loss') else 'N/A',
                str(r.get('started_at', ''))[:19]
            ])
        
        rounds_table = Table(rounds_table_data, colWidths=[50, 80, 80, 70, 60, 140])
        rounds_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ]))
        elements.append(rounds_table)
    else:
        elements.append(Paragraph("No training rounds completed yet.", normal_style))
    
    elements.append(Spacer(1, 20))
    
    # ---- HOSPITALS SECTION ----
    elements.append(Paragraph("3. Participating Hospitals", heading_style))
    
    if hospitals_data:
        hosp_table_data = [["Hospital", "Location", "Data Size", "Status", "Registered"]]
        for h in hospitals_data:
            hosp_table_data.append([
                str(h.get('name', '')),
                str(h.get('location', 'N/A')),
                str(h.get('data_size', 0)),
                'Active' if h.get('is_active') else 'Inactive',
                str(h.get('registered_at', ''))[:19]
            ])
        
        hosp_table = Table(hosp_table_data, colWidths=[100, 100, 70, 60, 150])
        hosp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ]))
        elements.append(hosp_table)
    
    elements.append(Spacer(1, 20))
    
    # ---- AUDIT LOG SECTION ----
    elements.append(Paragraph("4. Audit Trail", heading_style))
    elements.append(Paragraph(
        "The following audit log records all system activities for compliance verification.",
        normal_style
    ))
    
    if audit_logs:
        audit_table_data = [["Timestamp", "Action", "Details"]]
        for log in audit_logs[:20]:  # Last 20 entries
            audit_table_data.append([
                str(log.get('timestamp', ''))[:19],
                str(log.get('action', '')),
                str(log.get('details', ''))[:60]
            ])
        
        audit_table = Table(audit_table_data, colWidths=[130, 120, 230])
        audit_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ]))
        elements.append(audit_table)
    
    elements.append(Spacer(1, 30))
    
    # ---- COMPLIANCE STATEMENT ----
    elements.append(Paragraph("5. Compliance Statement", heading_style))
    compliance_text = (
        "This federated learning system has been designed to comply with data protection "
        "regulations. Key compliance measures include:<br/><br/>"
        "• <b>Data Locality:</b> All patient data remains on-premises at each hospital.<br/>"
        "• <b>Differential Privacy:</b> Calibrated noise is added to model updates.<br/>"
        "• <b>Update Clipping:</b> Model updates are clipped to limit sensitivity.<br/>"
        "• <b>Audit Logging:</b> All system actions are recorded with timestamps.<br/>"
        "• <b>Access Control:</b> Hospital nodes authenticate via unique API keys.<br/>"
        "• <b>Secure Communication:</b> All data transmitted via encrypted channels.<br/><br/>"
        "No individually identifiable patient information can be reconstructed from "
        "the shared model weights."
    )
    elements.append(Paragraph(compliance_text, normal_style))
    
    # Build PDF
    doc.build(elements)
    
    return output_path


def generate_csv_report(
    rounds_data: List[dict],
    output_path: str = "reports/training_report.csv"
) -> str:
    """Generate CSV report of training rounds"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow([
            'Round', 'Status', 'Participants', 'Target',
            'Global Accuracy', 'Global Loss', 'Started At', 'Completed At'
        ])
        
        # Data rows
        for r in rounds_data:
            writer.writerow([
                r.get('round_number', ''),
                r.get('status', ''),
                r.get('num_participants', ''),
                r.get('target_participants', ''),
                f"{r['global_accuracy']*100:.2f}%" if r.get('global_accuracy') else '',
                f"{r['global_loss']:.4f}" if r.get('global_loss') else '',
                str(r.get('started_at', ''))[:19],
                str(r.get('completed_at', ''))[:19] if r.get('completed_at') else ''
            ])
    
    return output_path


def generate_audit_csv(
    audit_logs: List[dict],
    output_path: str = "reports/audit_log.csv"
) -> str:
    """Generate CSV of audit logs"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['Timestamp', 'Action', 'Details', 'Hospital ID', 'IP Address'])
        
        for log in audit_logs:
            writer.writerow([
                str(log.get('timestamp', '')),
                log.get('action', ''),
                log.get('details', ''),
                log.get('hospital_id', ''),
                log.get('ip_address', '')
            ])
    
    return output_path