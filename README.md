# collusive-agents

*Collusion Without Text: Task Claims as a Channel Between LLM Agents*
(Apart Research AI Collusion Sprint 2026, Track 3.1).

Four LLM agents claim tasks on a shared Jira-style board, one claim each. Two of them are
a secret coalition, and their goal is a separation-of-duties breach. The text channel is
cut, and the experiment varies what agents can see of earlier claims.

## Setup

```bash
uv sync
cp .env.example .env    # add the API keys or endpoint you use
```

## Run

```bash
# Check the pipeline with a fake API (no key, no cost)
uv run python -m action_carriers.run --dry-run --seeds 2 --out /tmp/dry

# A run as in the paper: 40 seeds x 6 conditions
uv run python -m action_carriers.run --provider modal --model gpt-oss-120b \
    --reasoning-effort medium --seeds 40 --out runs
```

Providers: `modal`, `groq`, `openrouter`. See `python -m action_carriers.run --help`.
`scripts/modal_endpoint.py` deploys and manages a Modal Dedicated Endpoint.

## Credits

The task environment ports Terrarium's `JiraTicketEnvironment` (`terrarium-agents`
0.1.1, MIT License), the environment used by Colosseum (Nakamura et al., 2026). See
`THIRD_PARTY_NOTICES`.
