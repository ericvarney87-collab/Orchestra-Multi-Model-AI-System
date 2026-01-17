import subprocess

class StockfishEngine:
    """Interface to Stockfish chess engine via UCI protocol"""
    
    def __init__(self, stockfish_path="/usr/games/stockfish"):
        self.stockfish_path = stockfish_path
        self.process = None
        self.available = self._check_available()
    
    def _check_available(self):
        """Check if Stockfish is installed"""
        try:
            result = subprocess.run([self.stockfish_path, "--help"], 
                                  capture_output=True, 
                                  timeout=1)
            return True
        except:
            return False
    
    def start(self):
        """Start Stockfish process"""
        if not self.available:
            raise Exception("Stockfish not found")
        
        self.process = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._send_command("uci")
        self._wait_for("uciok")
    
    def _send_command(self, command):
        """Send UCI command to Stockfish"""
        if self.process:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
    
    def _wait_for(self, expected):
        """Wait for specific response"""
        while True:
            line = self.process.stdout.readline().strip()
            if expected in line:
                return line
    
    def analyze_position(self, fen, depth=20):
        """
        Analyze a chess position
        Args:
            fen: Position in FEN notation
            depth: Analysis depth (default 20)
        Returns:
            dict with best_move, evaluation, pv
        """
        if not self.process:
            self.start()
        
        self._send_command(f"position fen {fen}")
        self._send_command(f"go depth {depth}")
        
        best_move = None
        evaluation = None
        pv = []
        
        while True:
            line = self.process.stdout.readline().strip()
            
            if "bestmove" in line:
                best_move = line.split()[1]
                break
            
            if "score cp" in line:
                parts = line.split()
                cp_index = parts.index("cp") + 1
                evaluation = int(parts[cp_index]) / 100.0
            elif "score mate" in line:
                parts = line.split()
                mate_index = parts.index("mate") + 1
                evaluation = f"Mate in {parts[mate_index]}"
            
            if "pv" in line:
                parts = line.split()
                pv_index = parts.index("pv") + 1
                pv = parts[pv_index:pv_index+5]
        
        return {
            "best_move": best_move,
            "evaluation": evaluation,
            "principal_variation": " ".join(pv)
        }
    
    def close(self):
        """Shutdown Stockfish"""
        if self.process:
            self._send_command("quit")
            self.process.wait(timeout=2)
            self.process = None
