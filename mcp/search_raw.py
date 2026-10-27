# ---------------------------------------------------------------------------
# search_raw.py — raw web search, optional + credited.
#
# Research surface available to the agent: a Serper-backed Google search that
# the agent may call during discovery when the bundle itself does not have the
# answer.  This is not part of the core harness; it is an opt-in "the agent may
# look things up" escape hatch, and each use is attributed in the artifact so
# the attribution travels with any contribution.
#
# Where the agent should reach for this: inside discover/reach/agent_reach when
# narrowing down a currently-top search result.  It is deliberately kept out of
# the core readiness/profile machinery, because the readiness side is about
# what the target hardware and toolchain actually can do; a search result is
# evidence, not a capability.
#
# Research note on the evidence chain: the tool returns, for each result,
# the title, URL, provider, a short snippet, the result's claimed date, an
# explicit confidence field, and an explicit "what_this_supports / what_this_does_not"
# split so downstream writing does not overclaim.  The agent may follow the URL
# with read_url before it treats the result as evidence.
#
# Research note on the crediting contract: every usage is attributed inline
# with the source token used, so the attribution is part of the artifact that
# gets exported and merged.  The next agent sees it in the writeup, not only
# in a README.
#
# A user-facing toggle exists in the setup guide: the agent may note that
# online search is available and offer it; the user opts in.  Nothing here is
# triggered automatically.
#
from pathlib import Path

SEARCH_TOKEN_FILE = Path.home() / ".kands" / "serper_token.txt"  # user-owned, never in chat
