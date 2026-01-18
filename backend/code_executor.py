import re
import subprocess
import tempfile
import webbrowser
from pathlib import Path

class CodeExecutor:
    """
    Handles execution of code blocks from Orchestra responses.
    
    SECURITY: Code is NEVER auto-executed. Always requires explicit user confirmation.
    """
    
    def detect_code_blocks(self, text):
        """Extract code blocks from markdown-style responses"""
        pattern = r'```(\w*)\s*\n?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [(lang.lower() if lang else 'html', code.strip()) for lang, code in matches]
    
    def execute_python(self, code, timeout=10):
        """
        Execute Python code with timeout.
        
        WARNING: This should ONLY be called after explicit user confirmation.
        """
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
    
    def process_response(self, response_text, auto_execute=False):
        """
        Detect code blocks in response and return them for user review.
        
        SECURITY: Never auto-executes. Requires explicit user confirmation.
        
        Args:
            response_text: The LLM response text
            auto_execute: DEPRECATED - Kept for API compatibility but ignored for security
        
        Returns:
            List of code blocks found, NOT executed automatically
        """
        code_blocks = self.detect_code_blocks(response_text)
        results = []
    
        for lang, code in code_blocks:
            # SECURITY: Return code for user review, don't execute
            results.append({
                'type': lang,
                'code': code,
                'language': lang,
                'requires_confirmation': True,
                'warning': '⚠️ This code will execute on your system with your user permissions. Review carefully before running.',
                'preview': code[:200] + ('...' if len(code) > 200 else '')
            })
    
        return results
    
    def execute_with_confirmation(self, lang, code, user_confirmed=False):
        """
        Execute code ONLY after explicit user confirmation.
        
        SECURITY: This is the ONLY way code should be executed.
        
        Args:
            lang: Language type (python, html, javascript)
            code: The code to execute
            user_confirmed: Must be explicitly set to True to execute
        
        Returns:
            Execution result or error
        """
        # CRITICAL: Require explicit confirmation
        if not user_confirmed:
            return {
                'success': False,
                'error': 'User confirmation required',
                'message': 'Code execution requires explicit user approval for security. User must review and confirm.'
            }
        
        # Log execution for audit trail
        print(f"[SECURITY] User confirmed execution of {lang} code ({len(code)} chars)")
        
        if lang in ['html', 'javascript', 'js']:
            # For HTML/JS, wrap if needed
            if not code.strip().startswith('<'):
                code = f'<html><body><script>{code}</script></body></html>'
            filepath = self.save_and_open_html(code)
            return {
                'success': True,
                'type': 'html',
                'file': filepath,
                'message': f'HTML saved to {filepath}'
            }
        
        elif lang == 'python':
            # Execute Python code
            exec_result = self.execute_python(code)
            
            # Create HTML page with Python output
            output_html = f"""
            <html>
            <head>
                <title>Python Output</title>
                <style>
                    body {{ 
                        font-family: monospace; 
                        padding: 20px; 
                        background: #1e1e1e; 
                        color: #d4d4d4; 
                    }}
                    .warning {{
                        background: #5a1e1e;
                        padding: 10px;
                        border-left: 4px solid #f48771;
                        margin-bottom: 20px;
                    }}
                    pre {{
                        background: #252526;
                        padding: 15px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                    .stderr {{
                        color: #f48771;
                    }}
                </style>
            </head>
            <body>
                <div class="warning">
                    ⚠️ Code executed with your user permissions
                </div>
                <h2>Python Execution Result</h2>
                <h3>Code:</h3>
                <pre>{code}</pre>
                <h3>Output:</h3>
                <pre>{exec_result.get('stdout', '(no output)')}</pre>
                {f'<h3>Errors:</h3><pre class="stderr">{exec_result.get("stderr", "")}</pre>' if exec_result.get('stderr') else ''}
                <p><small>Return code: {exec_result.get('returncode', 'N/A')}</small></p>
            </body>
            </html>
            """
            filepath = self.save_and_open_html(output_html)
            return {
                'success': True,
                'type': 'python',
                'file': filepath,
                'result': exec_result,
                'message': f'Python code executed. Output saved to {filepath}'
            }
        
        else:
            return {
                'success': False,
                'error': f'Unsupported language: {lang}',
                'message': f'Language "{lang}" is not supported for execution'
            }


# Example usage showing secure pattern:
"""
# In your Flask API or main code:

executor = CodeExecutor()

# 1. Detect code blocks in LLM response
response_text = "Here's some code: ```python\nprint('hello')\n```"
code_blocks = executor.process_response(response_text)

# 2. Show to user in UI for review
for block in code_blocks:
    print(f"Found {block['language']} code:")
    print(block['warning'])
    print(block['preview'])
    
    # 3. User must explicitly confirm
    user_input = input("Execute this code? (yes/no): ")
    
    if user_input.lower() == 'yes':
        # 4. Only execute with confirmed=True
        result = executor.execute_with_confirmation(
            block['language'],
            block['code'],
            user_confirmed=True  # MUST be explicit
        )
        print(result)
    else:
        print("Code execution cancelled by user")
"""
