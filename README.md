# MoodTune — Persuasive AI Music Advisor

A web-based AI chatbot that recommends mood-appropriate music to
university students, comparing **persuasive** versus **neutral**
communication styles in a controlled A/B experiment.

Built as the major artefact for **MSc Applied Artificial
Intelligence and User Experience** at Abertay University
(graduating August 2026).

> ⚠️ This repository is published for academic portfolio review only.
> The code and research methodology are protected — see [LICENSE](LICENSE).

---

## Research question

**Does an AI music recommender that uses warm, persuasive language
(grounded in Cialdini's Six Principles of Influence and the Fogg
Behaviour Model) produce a greater positive change in mood,
enjoyment, and intention to use, compared to a neutral information-only
recommender?**

## Study design at a glance

A six-step within-subject experiment, randomised at consent:

1. **Consent** — full GDPR-compliant participant information sheet
2. **Pre-test** — demographics, baseline mood / focus / stress
3. **Mood induction** — Velten (1968) statements paired with calibrated calm imagery (Russell's Circumplex Model, 1980)
4. **Chatbot interaction** — randomised to **Regular** or **Persuasive** condition
5. **Post-test** — mood / focus / stress, enjoyment, trust, usability, intention to use, manipulation check
6. **Debrief** — full experimental disclosure plus a feel-good restoration playlist

Withdrawal is available on every screen via a persistent visible
button (BPS Code of Ethics 2021; GDPR Article 7(3)).

## Technology stack

- **Python 3.11** + **Streamlit** for the web app
- **Groq Cloud API** running **LLaMA 3.3 70B** for natural-language understanding and music recommendation generation
- **Spotify Web API** (via Spotipy) for real track resolution and preview audio
- **Spotify Audio Features** as an *applied AI validation layer* — every recommended track is scored against a quantitative mood-appropriate target range for valence, energy, danceability, and tempo
- **pandas** for structured data capture and analysis

## Applied AI components

| # | Component | Function |
|---|---|---|
| 1 | LLaMA 3.3 70B (Groq) | Natural-language mood detection and recommendation generation |
| 2 | Spotify Audio Features | Pre-trained audio ML models verify recommendations are *acoustically* appropriate, not just lyrically appropriate |
| 3 | Mood taxonomy (12 profiles) | Rule-based AI validation layer mapping natural-language mood to quantitative audio-feature targets |
| 4 | Multi-language preference extraction | Regex-based artist and genre detection across English plus Nigerian, Arabic, Nordic, East Asian, and South Asian musical traditions |

## UX design rationale

Every UX choice maps to a published source:

- **Dark theme + purple accent** — Bonnardel et al. (2011): music apps with dark UI score higher on perceived quality
- **Step progress indicator** — Tuch et al. (2009): users feel in control and less anxious in multi-step processes
- **Card-based layout** — Nielsen Heuristic #6: Recognition over Recall
- **Audio-feature transparency bars** — Nielsen Heuristic #1: Visibility of System Status
- **Co-equal image-and-text mood induction** — Rottenberg, Ray & Gross (2007): neither stimulus dominates

## Persuasion theory

The Persuasive condition operationalises:

- **Cialdini (1984) — Six Principles of Influence:** Liking, Social Proof, Authority
- **Fogg Behaviour Model (2009):** Persuasion = Motivation × Ability × Trigger

The Regular condition deliberately strips all of the above to create a clean independent-variable contrast.

## Safety and ethics

- Two-step withdrawal confirmation, available on every active screen
- Refresh-proof routing via hashed-token URL parameters, so a browser refresh never leaks the participant's experimental condition
- Crisis-response detection layer
- Content safety rules in every system prompt
- Velten induction is mild only and post-debrief includes a validated mood-restoration step

## Running the app locally

> Note: This code is published for portfolio review. To run it,
> you will need your own Groq and Spotify API credentials.

1. Clone:
```bash
   git clone https://github.com/noble-chidera-onyema/moodtune.git
   cd moodtune
```
2. Virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate
   venv\Scripts\activate
```
3. Install:
```bash
   pip install -r requirements.txt
```
4. Copy `.env.example` to `.env` and add:
   - `GROQ_API_KEY` from https://console.groq.com
   - `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` from https://developer.spotify.com/dashboard
5. Run:
```bash
   streamlit run app.py
```

## Project structure

moodtune/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Template for API keys
├── .gitignore          # Excludes secrets and participant data
├── LICENSE             # All Rights Reserved
└── README.md           # This file


## Academic references

- Cialdini, R. B. (1984). *Influence: The Psychology of Persuasion*.
- Fogg, B. J. (2009). A behavior model for persuasive design. *Persuasive '09*.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6).
- Velten, E. (1968). A laboratory task for induction of mood states. *Behaviour Research and Therapy*, 6(4).
- Nielsen, J. (1994). Heuristic evaluation. In *Usability Inspection Methods*.
- Bonnardel, N., Piolat, A., & Le Bigot, L. (2011). The impact of colour on website appeal. *Displays*, 32(2).
- Tuch, A. N. et al. (2009). Visual complexity of websites. *International Journal of Human-Computer Studies*, 67(9).
- Bai, H. et al. (2025). LLM-generated messages can persuade humans on policy issues. *Nature Communications*, 16, 6037.

## Author

**Noble Chidera Onyema**
MSc Applied Artificial Intelligence and User Experience,
Abertay University (graduating August 2026)

[LinkedIn](https://linkedin.com/in/noble-chidera-onyema-1a88b53ab) ·
onyemanoble1628@gmail.com

## License

**All Rights Reserved.** See [LICENSE](LICENSE) for full terms.
This repository is published for academic portfolio review only.
For collaboration or licensing enquiries, contact the author.
