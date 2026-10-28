# Rally

Volleyball highlight reels and team stats from your own film. Upload practice or match video, name the players once, confirm the touches the AI finds, and get:

- a highlight reel for any player
- a team stats site teammates can open on their phones, where they can tap their name and watch every touch they made

**Team site:** https://labelerguy.github.io/Rally/

## How it works

1. **Detect.** A pose model finds every person in each frame and tracks them.
2. **Group.** At practice nobody wears jerseys, so Rally groups people by what they're wearing (shirt and shorts color). Tracks that show up in the same frame are never merged, because one person can't be in two places at once.
3. **Find touches.** Every contact has a body signature: a jump with an arm above the head (attack), both hands up together (set or block), forearms together low in front (pass). Rally flags each one as a candidate. It doesn't track the ball, because the ball is a few pixels wide on a phone camera from the baseline.
4. **You confirm.** In the review app, you name each group once, then press `K` to keep or `X` to drop each candidate, and tag the result. Nothing counts toward stats or reels until a person keeps it.
5. **Reels and stats** come only from confirmed touches.

The AI does the searching. People make the calls only they can make.

## Setup (Mac)

```bash
git clone https://github.com/labelerguy/Rally && cd Rally
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

On Apple Silicon, Rally runs on the GPU automatically.

## Use it

```bash
# 1. Process a video. Plan on about 20 to 40 minutes for 2 hours of film on an M-series Mac.
python -m rally process ~/Movies/practice-10-27.mp4 --name 2026-10-27-practice

#    To try a slice first:
python -m rally process ~/Movies/practice-10-27.mp4 --name test --start 20:00 --end 30:00

# 2. Review: opens in your browser. Tab 1 names the players. Tab 2 confirms touches.
python -m rally review 2026-10-27-practice

# 3. Export a reel (or use the button in the review app)
python -m rally reel 2026-10-27-practice --player "Sir"
python -m rally reel 2026-10-27-practice --player "Sir" --actions attack,block --outcomes great

# 4. Publish stats to the team site
python -m rally publish 2026-10-27-practice --push
```

### Review app shortcuts

| Key | Action |
|---|---|
| `K` / `Enter` | Keep (it's a real touch by this person) |
| `X` / `Backspace` | Not a touch |
| `1`–`7` | Skill: attack, set, pass, dig, block, serve, other |
| `Q` `W` `E` | Result: great, in play, error |
| `Space` | Replay the clip |
| `↑` `↓` | Previous / next |
| `A` | Add a touch the AI missed at the current video time |

**Wrong person?** Use the "Who is this?" picker. **Fast pass:** set the filter to one player and the confidence floor to about 0.5, then sort by "Most likely first."

### Letting teammates watch clips on the site

Paste the session's YouTube link into the review app header before publishing. The site then plays each player's touches straight from YouTube, so no video gets hosted here. An unlisted YouTube upload works.

## Filming tips that matter most

1. **Highest resolution you can.** A 360p YouTube copy makes far-court players about 40 pixels tall. Get the original file from whoever filmed it, since 1080p or 4K works much better.
2. **Tripod behind the end line, high up.** Balltime and Hudl ask for the same setup. A low sideline angle hides people behind each other at the net.
3. **Don't move the camera mid-session.** Rally handles it, but every move breaks tracks.
4. **Distinct practice shirts help grouping** the way jerseys help in matches.

## What v1 doesn't do yet

- **Ball tracking.** Touches come from body pose, so expect false candidates (stretching, high-fives) and some misses. Review handles both.
- **Jersey-number reading for matches.** Grouping by clothing fails when the whole team wears the same jersey. Next version: OCR the numbers.
- **Rotations, rally win/loss, and side-out %.** These need court mapping and point tracking.

## Data

Video never goes to GitHub. Everything heavy lives in `work/<session>/` on your machine (gitignored). Only aggregated stats and clip timestamps go in `docs/data/`.

Because the site is public, anyone with the link can see names and stats. Use first names or nicknames if that matters to the team.
