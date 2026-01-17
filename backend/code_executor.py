import re
import subprocess
import tempfile
import webbrowser
from pathlib import Path

class CodeExecutor:
    """Handles execution of code blocks from Orchestra responses"""
    
    def detect_code_blocks(self, text):
        """Extract code blocks from markdown-style responses"""
        pattern = r'```(\w*)\s*\n?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [(lang.lower() if lang else 'html', code.strip()) for lang, code in matches]
    
    def execute_python(self, code, timeout=10):
        """Execute Python code safely with timeout"""
        try:
            result = subprocess.run(
                ['python3', '-c', code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Execution timeout (10s limit)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def save_and_open_html(self, code):
        """Save HTML/JS to temp file (don't auto-open)"""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.html',
            delete=False,
            dir='/tmp'
        )
        temp_file.write(code)
        temp_file.close()
    
        filepath = temp_file.name
        # Return filepath - let frontend handle opening
        return filepath
    
    def process_response(self, response_text):
        """Detect and execute code blocks in response"""
        code_blocks = self.detect_code_blocks(response_text)
        results = []
    
        for lang, code in code_blocks:
            if lang in ['html', 'javascript', 'js']:
                # For HTML/JS, wrap if needed
                if not code.strip().startswith('<'):
                    code = f'<html><body><script>{code}</script></body></html>'
                filepath = self.save_and_open_html(code)
                results.append({'type': 'html', 'file': filepath})  # Just return filepath
        
            elif lang == 'python':
                exec_result = self.execute_python(code)
                # Create HTML page with Python output
                output_html = f"""
                <html>
                <head><title>Python Output</title>
                <style>body {{ font-family: monospace; padding: 20px; background: #1e1e1e; color: #d4d4d4; }}</style>
                </head>
                <body>
                <h2>Python Execution Result</h2>
                <pre>{exec_result.get('stdout', '')}</pre>
                {f'<pre style="color: #f48771;">{exec_result.get("stderr", "")}</pre>' if exec_result.get('stderr') else ''}
                </body>
                </html>
                """
                filepath = self.save_and_open_html(output_html)
                results.append({'type': 'python', 'file': filepath})
    
        return results
