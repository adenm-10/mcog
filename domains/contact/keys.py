# domains/contact/keys.py
"""THE one definition of the INTERFACE/TASK split, and nothing else.

Deliberately dependency-free (no numpy, no pymunk, no gymnasium) so every
consumer can import it: eval_contact, the scorers, the probes, the floor
builders and test_code all need this list and none of them should have to pull
in the simulator to get it.

WHY IT LIVES IN ONE PLACE. This list used to be copy-pasted into five modules
with a `static` gate asserting the copies agreed. The gate worked, but it only
ever fired AFTER a copy drifted, and keeping five literals in step is the same
shape as the six other "same quantity computed twice" defects this project has
paid for (obs() vs her_buffer's divisor, step() vs HER's arrival, the launcher's
protocol vs the scorer's). One definition cannot drift.

INTERFACE keys change what the policy's outputs and inputs MEAN. They are
EXCLUDED from the env digest, so two arms differing only here are directly
comparable on one benchmark -- which is the entire reason obs v1 vs v2 and
face vs count can be arms rather than separate experiments.

TASK keys change what success IS, or which states are visited. They stay INSIDE
the digest. Two arms differing in a task key are two experiments and need a
transfer eval before their numbers can be compared.

Adding a key here silently REMOVES it from the digest and orphans every stored
score. Do not add one without deciding which side of that line it is on.
"""
from __future__ import annotations

#: Read per cell from the run's own metadata; never in the digest.
IFACE_KEYS = (
    "action_interface",
    "slip_model",
    "slip_limit",
    "restrict_contact_actions",
    "mask_inactive_finger",
    "gap_assist",
    "obs_version",
    #: D1: whether xi's Gamma slot carries the 4-way contact FACE or the 3-way
    #: contact COUNT. It changes what the policy is TOLD, not what the task is
    #: -- enforcing a count is `guard_contact_count`, which is a TASK key.
    "xi_gamma_mode",
    "omega_max_rad_s",
    "force_scale_kgcms2",
    "normalize_goal_keys",
    "rl_algo",
)

#: Env kwargs omitted from the digest WHILE AT THEIR DEFAULT, so that adding one
#: does not rehash every archived config. ONLY safe for a key whose default
#: reproduces the old behaviour bit-identically -- verify by replaying a known
#: digest, never by assuming.
#: `xi_gamma_mode` is deliberately NOT here: it is an INTERFACE key, so the
#: digest already excludes it at every value, not just its default.
STAMP_OMIT_IF_DEFAULT = {
    "push_spawn_along_frac": None,
    "guard_contact_count": None,
    #: Only ever READ under guard_object_still="displacement", which no archived
    #: run sets, so at its default it is inert in every stored config. Verified
    #: by replaying 249434216cd2, not assumed.
    "guard_disp_eps_cm": 2.0,
}
