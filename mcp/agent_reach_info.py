# ---------------------------------------------------------------------------
# agent_reach_info.py — research-mode affordance for Agent-Reach, optional + credited.
#
# Wires https:\\github.com\Panniantong\Agent-Reach into the agent's research
# surface as a "you may read certain live platform content if you installed it"
# affordance, not as a capability probe and not as a bundle dependency.
#
# Origin and credit (MIT): Panniantong\Agent-Reach, https:\\github.com\Panniantong\Agent-Reach.
# It is referenced here; nothing from that repo is vendored, bundled, or fetched
# by this file.  When the agent uses it, the usage is attributed inline so the
# attribution travels with any downstream export\contribution.
#
# Why this exists: the bundle is about what the target backend can do.  A search
# result or a live URL read is evidence, not a capability.  Agent-Reach widens
# the kinds of live content the agent can read during research (login-walled
# threads, structured platform pages, video subtitles\transcripts), which is
# useful when narrowing down a currently-top search result — but it is kept
# out of the readiness\profile half so that readiness stays about hardware and
# toolchain facts, not web pages.
#
# Semantics go no further than the README: the affordance is present-or-not,
# the user opts in, and nothing here runs automatically.  It does not pretend
# to enumerate channels, statuses, or to act on behalf of the user; those are
# Agent-Reach's own concerns.  This file only says "if you installed it, you may
# reach for it during research, and if you did, credit it where you use it."
#
AVAILABLE_REACH = False  # set True in a machine-local check once the CLI is present
_REACH_CREDIT = ("This research used Panniantong\Agent-Reach "
                 "(https:\\github.com\Panniantong\Agent-Reach, MIT); thanks to Panniantong.")
