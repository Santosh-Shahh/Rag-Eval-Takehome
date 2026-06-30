import markdown
from xhtml2pdf import pisa
from pathlib import Path

def convert_md_to_pdf(md_path, pdf_path):
    # Read Markdown
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert to HTML
    html_content = markdown.markdown(
        md_content, 
        extensions=['extra', 'codehilite', 'tables', 'fenced_code']
    )

    # Wrap in standard HTML structure with custom CSS styles
    styled_html = f"""
    <html>
    <head>
    <style>
        @page {{
            size: letter;
            margin: 0.8in;
        }}
        body {{
            font-family: Helvetica, Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.4;
            color: #333;
        }}
        h1, h2, h3, h4 {{
            color: #1a365d;
            font-family: Helvetica-Bold, Arial, sans-serif;
        }}
        h1 {{ font-size: 18pt; margin-top: 20px; border-bottom: 1px solid #1a365d; padding-bottom: 5px; }}
        h2 {{ font-size: 14pt; margin-top: 15px; border-bottom: 0.5px solid #4a5568; padding-bottom: 3px; }}
        h3 {{ font-size: 11pt; margin-top: 10px; }}
        code {{
            font-family: Courier, monospace;
            background-color: #f7fafc;
            padding: 2px 4px;
            font-size: 9pt;
        }}
        pre {{
            background-color: #f7fafc;
            border: 1px solid #e2e8f0;
            padding: 10px;
            font-family: Courier, monospace;
            font-size: 8.5pt;
            white-space: pre-wrap;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            margin-bottom: 15px;
        }}
        th, td {{
            border: 1px solid #cbd5e0;
            padding: 6px 8px;
            text-align: left;
        }}
        th {{
            background-color: #edf2f7;
            font-weight: bold;
            color: #2d3748;
        }}
        blockquote {{
            border-left: 4px solid #cbd5e0;
            padding-left: 10px;
            margin-left: 0;
            color: #4a5568;
            font-style: italic;
        }}
    </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate PDF
    with open(pdf_path, "w+b") as out_file:
        pisa_status = pisa.CreatePDF(styled_html, dest=out_file)
    
    if pisa_status.err:
        print("Error during PDF generation!")
    else:
        print(f"Successfully generated PDF: {pdf_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    convert_md_to_pdf(base_dir / "submission.md", base_dir / "submission.pdf")
