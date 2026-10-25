"""Literature-informed noise rules; clinical relevance remains a downstream task.

Full-message matching only. No drug lexicon, first-person requirement, symptom
dictionary, username guesses, cross-ID text deletion, or minimum-length veto.
"""
import argparse
import json
from pathlib import Path
import re
import unicodedata

from . import reddit_basic as basic

PLACEHOLDER = re.compile(
    r'\[\s*(?:deleted(?: by (?:user|author))?|removed(?: by (?:reddit|moderators?|mods?|admin))?'
    r'(?: on account of violating the content policy\.?)?)\s*\]', re.I)
REDACTED = re.compile(
    r'(?:this (?:post|comment) (?:was|has been) (?:mass )?deleted and anonymized with redact'
    r'|this (?:post|comment) (?:was|has been) (?:mass )?deleted using redact)\.?', re.I)
MENTION = re.compile(r'(?<!\w)/?[ur]/[\w-]+\b', re.I)
LINK_REDIRECT = re.compile(
    r'(?:here(?:\s+is|\s*\x27s)?(?:\s+the)?\s+link|link|source|read (?:this|here)|'
    r'check (?:this|it) out|more (?:info|information)(?: here)?|see (?:this|here))'
    r'\s*[:.!-]?\s*(?:<url>\s*)+[.!]*', re.I)
RECRUITMENT = re.compile(
    r'(?:please\s+)?(?:take|complete|fill (?:in|out))\s+(?:(?:my|our|this|the|a)\s+)?'
    r'(?:short\s+|quick\s+)?(?:survey|questionnaire)\s*[:.!-]?\s*(?:<url>\s*)+[.!]*', re.I)
# Only these explicitly enumerated decorations may be ignored for a social
# full-match. Never strip all emoji: "thanks 🤢" must retain the symptom signal.
SOCIAL_DECORATION = set('😊🙂😀😃😄😁☺❤♥🤍💙💚💛💜🧡🩷🩵🤎🖤🎉🎊👏🙌👍🙏🎈💪🤩✨⭐\ufe0f\u200d')


def social_surface(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('’', "'")
    text = ''.join(' ' if c in SOCIAL_DECORATION else c for c in text)
    return ' '.join(text.strip(' \n\r\t.!').split())


def terminal_automation(feature):
    """Self-disclosure at the end of an unquoted comment, not any 'bot' mention."""
    segments = [s for s in feature['segments'] if s['text'].strip()]
    if feature.get('kind') != 'comments' or not segments:
        return False
    if any(s['role'] != 'prose' or s.get('field') != 'body' for s in segments):
        return False
    last = ' '.join(segments[-1]['text'].split())
    if basic.template_only(last):
        return True
    return bool(len(segments) > 1 and basic.CONTACT.fullmatch(last) and
                basic.template_only(segments[-2]['text']))


def prepare(feature, policy):
    result = basic.prepare(feature, policy)
    # New decisions apply to original retained segments; base segment indexes
    # and full source remain available even for whole-message exclusions.
    if result['decision'] != 'candidate':
        return result
    segments = result['segments']
    substantive = [s for s in segments if s['text'].strip() and
                   not (s['role'] == 'prose' and PLACEHOLDER.fullmatch(s['text'].strip()))]
    removed_placeholders = len(substantive) != len(segments)
    reason = None
    if not substantive:
        reason = 'expanded_deleted_placeholder'
    elif policy.get('terminal_automation', False) and terminal_automation(feature):
        reason = 'terminal_automation_disclosure'
    else:
        prose_only = all(s['role'] == 'prose' for s in substantive)
        text = '\n\n'.join(s['text'] for s in substantive)
        compact = ' '.join(text.split())
        if prose_only and REDACTED.fullmatch(compact):
            reason = 'redaction_notice_only'
        elif prose_only and social_surface(text) in set(policy.get('social_templates', [])):
            reason = 'social_template_only'
        elif prose_only and not basic.readable(MENTION.sub(' ', text)):
            reason = 'link_or_mention_only'
        elif prose_only:
            masked = basic.URL.sub('<url>', compact)
            if LINK_REDIRECT.fullmatch(masked):
                reason = 'link_redirect_only'
            elif RECRUITMENT.fullmatch(masked):
                reason = 'survey_link_only'
    if removed_placeholders and substantive:
        # Do not alter index-based segment provenance; mark these for downstream
        # omission when rendering. Title or independent body remains available.
        result['noise_placeholder_segments'] = [i for i, s in enumerate(segments)
            if s['role'] == 'prose' and PLACEHOLDER.fullmatch(s['text'].strip())]
    if reason:
        result.update(decision='exclude', reason=reason)
    result['noise_policy_revision'] = policy['revision']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('.cache/reddit-cleaning/v2-strict'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path('configs/reddit-basic-literature.json'))
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--max-records', type=int)
    args = parser.parse_args()
    report = basic.run(args.source_dir, args.output_dir, json.loads(args.policy.read_text()),
                       args.sample_size, args.seed, args.max_records,
                       preparer=prepare, supporting_code_paths=[__file__])
    print(json.dumps({k: report[k] for k in ('status', 'processed_records', 'dispositions', 'reasons')}, indent=2))


if __name__ == '__main__':
    main()
