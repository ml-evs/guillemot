# This file was edited with the assistance of an AI model and requires human review from the contributor.
"""Turning a TOPAS parse error into something the agent can act on.

The outputs quoted here are real: they are what TOPAS 6 returned during the FeSb
session, where the agent misread one of them and spent an extra remote run removing
an `@` from the wrong line.
"""

import textwrap

from guillemot.tools.topas import _inp_excerpt

# `scale @ 1.0E-4` is written on line 41 but reported as line 42: TOPAS names the
# point where its parser gave up, not the mistake.
SCALE_ERROR = textwrap.dedent(
    """\
    *** Error loading sstring_in
        at LINE 42
        See log file c:\\users\\group_user\\topas_6\\tc.log

    *** Error at: 1.0E-4
        Maybe an equations starting with an equal sign and
        ending in a semicolon is what is meant.
    """
)

# `Phase_Density_g_on_cm3( @ ... )` on line 43, reported exactly.
DENSITY_ERROR = SCALE_ERROR.replace("at LINE 42", "at LINE 43").replace(
    "*** Error at: 1.0E-4", "*** Error at: @"
)


def _inp_lines() -> str:
    body = [f"line {n}" for n in range(1, 41)]
    body += [
        "\t\tscale @  1.0E-4",
        "\t\tr_bragg  1.0",
        "\t\tPhase_Density_g_on_cm3( @  15.6 )",
        '\t\tOut_CIF_STR("FeSb.cif")',
    ]
    return "\n".join(body)


def test_reported_line_is_marked_and_quoted():
    excerpt = _inp_excerpt(_inp_lines(), SCALE_ERROR, "FeSb_riet_01.inp")

    assert "FeSb_riet_01.inp at line 42" in excerpt
    assert "on the token `1.0E-4`" in excerpt
    assert "> 42 | \t\tr_bragg  1.0" in excerpt


def test_the_real_culprit_is_marked_when_the_line_number_is_off_by_one():
    """The token is the reliable pointer when the line number is not."""
    excerpt = _inp_excerpt(_inp_lines(), SCALE_ERROR, "FeSb_riet_01.inp")

    assert "* 41 | \t\tscale @  1.0E-4" in excerpt


def test_a_single_character_token_is_named_but_not_marked():
    """`@` appears on half the lines of an .inp; marking them all is just noise."""
    excerpt = _inp_excerpt(_inp_lines(), DENSITY_ERROR, "FeSb_riet_02.inp")

    assert "on the token `@`" in excerpt
    assert "> 43 | \t\tPhase_Density_g_on_cm3( @  15.6 )" in excerpt
    assert "*" not in excerpt


def test_context_window_surrounds_the_reported_line():
    excerpt = _inp_excerpt(_inp_lines(), SCALE_ERROR, "x.inp", context=2)
    numbers = [
        line.split("|")[0].strip(" >*") for line in excerpt.splitlines() if "|" in line
    ]

    assert numbers == ["40", "41", "42", "43", "44"]


def test_window_is_clipped_at_the_ends_of_the_file():
    excerpt = _inp_excerpt("only one line\n", "at LINE 1\n", "x.inp")

    assert "> 1 | only one line" in excerpt


def test_no_line_number_means_no_excerpt():
    assert _inp_excerpt(_inp_lines(), "Abnormal program termination.", "x.inp") == ""


def test_a_line_number_past_the_end_of_the_file_is_ignored():
    """Better silence than an excerpt of a file the number cannot refer to."""
    assert _inp_excerpt(_inp_lines(), "    at LINE 9999\n", "x.inp") == ""


def test_unreadable_file_is_not_fatal():
    assert _inp_excerpt("", SCALE_ERROR, "x.inp") == ""


# --------------------------------------------------------------------------------------
# What a successful refinement changed, and the warnings about it
# --------------------------------------------------------------------------------------

REPORT_INP = """xdd p.xy
	bkg @ 20 0 0
	LP_Factor(90)
	Specimen_Displacement(displ, 0.0)
str
	a lpa 6.17
	b lpa 6.17
	c lpc 6.17
	al 90.
	be 90.
	ga 90.
	space_group 225
	site Pb1 x 0 y 0 z 0 occ Pb 1 beq @ 0.5
	scale @ 0.05
	CS_L(@, 0.30)
"""


def _out(**values):
    """REPORT_INP as TOPAS would write it back, with refined values marked by a backtick."""
    v = {
        "a": "6.12",
        "c": "6.12",
        "beq": "0.8",
        "cs": "150`",
        "displ": "-0.01",
    } | values
    return (
        REPORT_INP.replace("bkg @ 20 0 0", "bkg @ 25.1` -3.2` 1.1`")
        .replace("(displ, 0.0)", f"(displ,{v['displ']}`)")
        .replace("a lpa 6.17", f"a lpa  {v['a']}`")
        .replace("b lpa 6.17", f"b lpa  {v['a']}`")
        .replace("c lpc 6.17", f"c lpc  {v['c']}`")
        .replace("beq @ 0.5", f"beq @  {v['beq']}`")
        .replace("scale @ 0.05", "scale @  0.031`")
        .replace("CS_L(@, 0.30)", f"CS_L(@, {v['cs']})")
    )


def test_report_lists_changes_and_is_quiet_for_a_sound_refinement():
    from guillemot.tools.topas import refinement_report

    changes, warnings = refinement_report(REPORT_INP, _out())
    assert "a lpa 6.17  ->  a lpa 6.12" in changes
    assert (
        "Specimen_Displacement(displ, 0.0)  ->  Specimen_Displacement(displ,-0.01)"
        in changes
    )
    assert warnings == []


def test_report_warns_about_what_went_wrong():
    from guillemot.tools.topas import refinement_report

    _, warnings = refinement_report(
        REPORT_INP, _out(a="5.99", c="5.85", beq="-0.4", cs="0.301`_LIMIT_MIN_0.3")
    )
    text = "\n".join(warnings)
    assert "cubic but a, b, c refined to different values" in text
    assert "The cell moved a long way" in text and "c 6.17 -> 5.85, -5.2%" in text
    assert "Negative beq" in text
    assert "0.301 nm" in text and "minimum limit (0.3)" in text


def test_report_warns_about_a_missing_lp_factor_and_huge_beq():
    """A synchrotron fit without LP_Factor refined beq to ~7 Å² to make up for it."""
    from guillemot.tools.topas import refinement_report

    no_lp = REPORT_INP.replace("\tLP_Factor(90)\n", "")
    _, warnings = refinement_report(
        no_lp, _out(beq="6.97").replace("\tLP_Factor(90)\n", "")
    )
    text = "\n".join(warnings)
    assert "There is no LP_Factor" in text
    assert "beq of 6.97" in text
