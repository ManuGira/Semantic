# Wordle-Like Game Server Architecture

## Overview

Stateless game server using **FastAPI** + **JWT tokens** for session management. No database required.

## Architecture Principle

**Stateless Design**: All game state stored in JWT tokens on the client side.

```
Client → Server (with JWT) → Process → Server → Client (with new JWT)
```

## File Structure

```
src/
  server_game.py        # FastAPI backend server
static/
  index.html           # Frontend (HTML + JavaScript)
```

## How It Works

### 1. Session Flow

1. **First request**: No token → Server creates new game state → Returns JWT
2. **Subsequent requests**: Client sends JWT → Server decodes & validates → Updates state → Returns new JWT
3. **Token storage**: Client saves JWT in `localStorage`
4. **Token expiration**: Automatic at midnight (resets daily challenge)

### 2. Daily Challenge System

```python
def get_daily_word(date_str: str) -> str:
    hash_val = int(hashlib.sha256(date_str.encode()).hexdigest(), 16)
    return WORD_LIST[hash_val % len(WORD_LIST)]
```

- Same date = same hash = same word for all players
- Deterministic and reproducible
- No coordination needed between servers

### 3. JWT Token Structure

```json
{
  "date": "2026-01-08",
  "guesses": [{"guess": "CRANE", "hints": [...]}],
  "attempts": 1,
  "completed": false,
  "won": false,
  "exp": 1736294400
}
```

- **Signed with HMAC**: Prevents client tampering
- **Expires at midnight**: Forces daily reset
- **Self-contained**: No database lookups needed

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve frontend HTML |
| `/api/info` | GET | Get game metadata (date, word length, max attempts) |
| `/api/guess` | POST | Submit guess, returns hints + new token |
| `/api/reset` | POST | Reset game, returns fresh token |

## Game Logic (Customizable)

**Location**: `check_guess()` function in [server_game.py](../src/server_game.py)

Current implementation: Simple Wordle clone (5-letter word matching)

```python
def check_guess(guess: str, secret_word: str) -> dict:
    # YOUR GAME LOGIC HERE
    return {
        "guess": guess,
        "hints": [...],
        "is_correct": bool
    }
```

**To replace with your game:**
1. Modify `check_guess()` to return your custom hint structure
2. Update frontend JavaScript to render your hints
3. Adjust `max_attempts` and `word_length` as needed

## Frontend Features

- **Auto-save**: Progress stored in `localStorage`
- **Session restoration**: Reloads game state on page refresh
- **Color-coded feedback**: Green (correct), Yellow (present), Gray (absent)
- **Real-time stats**: Attempts counter, remaining tries

## Key Benefits

✅ **Scalable**: Any server can handle any request (no sticky sessions)  
✅ **Simple**: No database, no Redis, no session storage  
✅ **Secure**: JWT signature prevents state forgery  
✅ **Daily reset**: Token expiration handles it automatically  
✅ **Fair**: Everyone gets same challenge on same day  

## Running the Server

```bash
uv run src/server_game.py
```

Server starts at: http://127.0.0.1:8000

## Security Considerations

- **Change `SECRET_KEY`** in production
- Use environment variables for secrets
- Consider HTTPS in production
- Rate limiting recommended for `/api/guess` endpoint

## Future Enhancements

- [ ] Add difficulty levels
- [ ] Statistics tracking (win rate, average attempts)
- [ ] Leaderboard (requires database)
- [ ] Share results (emoji grid like Wordle)
- [ ] Multiple game modes
