def html_default ():

    FALLBACK_HTML_TEMPLATE = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Content</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 40px auto;
                padding: 0 20px;
                line-height: 1.6;
                color: #333;
            }}
            .warning {{
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 4px;
                padding: 10px 15px;
                margin-bottom: 20px;
                font-size: 0.85em;
                color: #856404;
            }}
            .content {{
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        </style>
    </head>
    <body>
        <div class="warning"> Displaying raw content (HTML generation failed after 3 attempts).</div>
        <div class="content">{content}</div>
    </body>
    </html>"""
    
    return FALLBACK_HTML_TEMPLATE

def html_chat_history():
    html = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Chat History</title>
        <style>
            body { font-family: Georgia, serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
            h1 { text-align: center; }
            .entry { border-bottom: 2px solid #eee; padding: 20px 0; }
            .timestamp { color: #999; font-size: 0.85em; }
            .user-query { background: #f0f4ff; padding: 10px 15px; border-radius: 8px; margin: 10px 0; }
            .ai-response { margin-top: 10px; }
        </style>
    </head>
    <body>
    <h1>Chat History</h1>
    """

    return html
