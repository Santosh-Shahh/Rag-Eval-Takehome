import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_security_pdf(output_path: Path):
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        spaceAfter=20
    )
    
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        spaceBefore=15,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        spaceAfter=8
    )
    
    # Title
    story.append(Paragraph("AetherDB Security Policy & Hardening Guide", title_style))
    story.append(Spacer(1, 10))
    
    # Section 1
    story.append(Paragraph("1. Network Security and Ports", h2_style))
    story.append(Paragraph(
        "AetherDB nodes expose three key ports: 8080 (REST Client API), 9090 (Raft Consensus), "
        "and 7070 (Gossip Protocol). Firewalls must be configured to prevent unauthorized external access. "
        "Specifically, Raft Consensus port 9090 and Gossip port 7070 must only be reachable by cluster members.",
        body_style
    ))
    
    # Section 2
    story.append(Paragraph("2. Authentication and Authorization", h2_style))
    story.append(Paragraph(
        "All data manipulation requests must be authenticated. The X-Aether-Token HTTP header "
        "is mandatory for accessing /api/v1/write and /api/v1/read. Health check requests "
        "on /api/v1/health do not require authentication.",
        body_style
    ))
    story.append(Paragraph(
        "Tokens should be cryptographically random and rotated every 30 days. Shared tokens "
        "across clients are strictly prohibited.",
        body_style
    ))
    
    # Section 3
    story.append(Paragraph("3. Encryption at Rest", h2_style))
    story.append(Paragraph(
        "To protect stored data from physical theft, encryption at rest must be enabled. "
        "AetherDB supports AES-256 block encryption for the WAL and SSTables. The encryption key "
        "must be managed using an external Key Management Service (KMS).",
        body_style
    ))
    
    doc.build(story)
    print(f"Generated PDF at: {output_path}")

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"
    os.makedirs(out_dir, exist_ok=True)
    generate_security_pdf(out_dir / "security_policy.pdf")
