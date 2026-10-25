"""Lossy strict-subset selection on top of an aggressive denoise version (x stays 20 underneath).

This layer is not denoising: it keeps the records with the most original information and drops the rest, which is
mostly normal discussion.  Every extra threshold is recorded in the manifest.  Information is measured without topic
terms: content words are the words left after URLs, generalized social phrases/words and English function words are
removed; "original" content comes only from sentences that are neither owned by an earlier kept record (copy) nor
shared by >= 3 kept records (template).  Promotion/solicitation is matched by structure only (no brand list) and
non-English records by function words.  No drug/symptom/first-person/topic/subreddit conditions, no random trimming.
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import polars as pl

from .reddit_denoise_aggressive import FUNCTION, SENTENCE, SOCIAL_PHRASES, SOCIAL_WORDS, order_key
from .reddit_denoise_refine import LEXICAL as LEX, URL

STOP = sorted(set(FUNCTION.split('|')) | set(SOCIAL_WORDS.split('|')))
REASONS = ('strict_promotion', 'strict_non_english', 'strict_repetitive', 'strict_mostly_borrowed', 'strict_info_below')
KEYS = {'strict_promotion': ('code_or_referral', 'referral_url', 'url_exclude', 'provider_listing', 'solicitation',
                             'promo_code_offer', 'giveaway', 'discount_offer', 'offer_handle', 'code_offer', 'code_exclude',
                             'app_builder', 'self_promo', 'prize_draw', 'service_offer', 'link_bio', 'promo_code', 'own_media',
                             'care_team_text', 'own_site', 'own_disclosure', 'obfuscated_code', 'glued_code', 'consult_free',
                             'referral_offer', 'profile_pointer', 'app_developer', 'creator_shop', 'photo_app_pitch',
                             'dev_beta', 'match_normalize', 'discount_link'),
        'strict_non_english': ('min_foreign_words', 'min_ratio_vs_english'),
        'strict_repetitive': ('min_content_words', 'max_top_share'),
        'strict_mostly_borrowed': ('min_content_words', 'max_original_share'),
        'strict_info_below': ('min_original_types',)}


def peak_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == 'darwin' else rss / 1024


def validate_policy(policy):
    rules = policy.get('strict', {})
    unknown = set(rules) - set(KEYS)
    if unknown or not rules:
        raise ValueError(f'strict rules must be a non-empty subset of {sorted(KEYS)}; unknown: {sorted(unknown)}')
    for name, rule in rules.items():
        if set(rule) != set(KEYS[name]):
            raise ValueError(f'{name} needs exactly {KEYS[name]}')
    return rules


def kept(version_dir, shard):
    """Records the given version keeps (decision == candidate) in one shard."""
    return pl.read_parquet(version_dir / 'decisions' / shard.name, columns=['rowid', 'decision']).filter(
        pl.col('decision') == 'candidate').select(pl.col('rowid').cast(pl.Int64))


def content_tokens(expr):
    """Lowercased content tokens of a text expression (URLs, social phrases/words and function words removed)."""
    rest = expr.str.to_lowercase().str.replace_all(URL, ' ').str.replace_all(rf'\b(?:{SOCIAL_PHRASES})\b', ' ')
    return rest.str.extract_all(LEX).list.eval(pl.element().filter(~pl.element().is_in(STOP)))


def sentence_rows(frame):
    """One row per distinct normalized sentence of each record (same normalization as the aggressive layer)."""
    return (frame.select('rowid', key=order_key(),
                         piece=pl.col('text').str.to_lowercase().str.replace_all(URL, ' ').str.extract_all(SENTENCE))
            .explode('piece')
            .with_columns(norm=pl.col('piece').str.replace_all(r'^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$', '').str.replace_all(r'\s+', ' '))
            .with_columns(sw=pl.col('norm').str.count_matches(LEX).cast(pl.UInt16), shash=pl.col('norm').hash(41, 43, 47, 53))
            .filter(pl.col('sw') > 0)
            .unique(subset=['rowid', 'shash'], keep='first', maintain_order=True))


def kept_sentence_stats(version_dir, work):
    """docs/owner of every sentence among the records the version keeps (owners removed earlier do not count)."""
    out = work / 'kept_sentence_stats.parquet'
    if not out.exists():
        keep = pl.concat([kept(version_dir, s) for s in sorted((version_dir / 'decisions').glob('part-*.parquet'))])
        (pl.scan_parquet(str(version_dir / 'work' / 'sentences' / '*.parquet'))
         .join(keep.lazy().select(pl.col('rowid').cast(pl.UInt32)), on='rowid', how='semi')
         .group_by('shash').agg(docs=pl.len().cast(pl.UInt32), owner=pl.col('key').min())
         .sink_parquet(out.with_suffix('.tmp')))
        out.with_suffix('.tmp').replace(out)
    return out


def record_features(frame, stats):
    """Per-record information and structure features for one shard of kept records."""
    toks = content_tokens(pl.col('text'))
    words = pl.col('text').str.to_lowercase().str.replace_all(URL, ' ').str.extract_all(LEX)
    base = frame.select(
        'rowid', 'kind', chars=pl.col('text').str.len_chars().cast(pl.UInt32), n_tok=words.list.len().cast(pl.Int32),
        url_n=pl.col('text').str.count_matches(URL).cast(pl.Int32),
        emoji_n=pl.col('text').str.count_matches(r'\p{Extended_Pictographic}').cast(pl.Int32))
    tc = (frame.select('rowid', t=toks).explode('t').drop_nulls('t').group_by('rowid', 't').len()
          .group_by('rowid').agg(content_n=pl.col('len').sum().cast(pl.Int32), content_types=pl.len().cast(pl.Int32),
                                 top=pl.col('len').max()))
    base = (base.join(tc, on='rowid', how='left')
            .with_columns(pl.col('content_n').fill_null(0), pl.col('content_types').fill_null(0),
                          top_share=(pl.col('top').fill_null(0) / pl.max_horizontal(pl.col('content_n'), pl.lit(1))).cast(pl.Float32))
            .drop('top'))
    s = (sentence_rows(frame).join(stats, on='shash', how='left')
         .with_columns(borrowed=(pl.col('owner').is_not_null() & (pl.col('owner') != pl.col('key')) & (pl.col('sw') >= 5))
                       | (pl.col('docs').fill_null(1) >= 3),
                       ct=content_tokens(pl.col('norm'))))
    orig = (s.filter(~pl.col('borrowed')).select('rowid', 'ct').explode('ct').drop_nulls('ct')
            .group_by('rowid').agg(original_n=pl.len().cast(pl.Int32), original_types=pl.col('ct').n_unique().cast(pl.Int32)))
    return (base.join(orig, on='rowid', how='left')
            .with_columns(pl.col('original_n').fill_null(0), pl.col('original_types').fill_null(0)))


def run_features(base_dir, version_dir, work):
    """Checkpointed per-shard features for all records the version keeps."""
    (work / 'features').mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stats = pl.read_parquet(kept_sentence_stats(version_dir, work))
    for shard in sorted((base_dir / 'candidates').glob('part-*.parquet')):
        out = work / 'features' / shard.name
        if out.exists():
            continue
        frame = (pl.read_parquet(shard, columns=['rowid', 'kind', 'created_utc', 'text'])
                 .join(kept(version_dir, shard), on='rowid', how='semi'))
        record_features(frame, stats).write_parquet(out.with_suffix('.tmp'), compression='zstd')
        out.with_suffix('.tmp').replace(out)
    return time.perf_counter() - started


PROMO_PARTS = ('code_or_referral', 'referral_url', 'provider_listing', 'solicitation', 'promo_code_offer', 'giveaway',
               'discount_offer', 'code_offer', 'app_builder', 'self_promo', 'prize_draw', 'service_offer', 'link_bio',
               'promo_code', 'own_media', 'care_team_text', 'own_site', 'own_disclosure', 'obfuscated_code', 'glued_code',
               'consult_free', 'referral_offer', 'profile_pointer', 'dev_role', 'dev_working_on', 'app_cta_link', 'app_beta',
               'creator_hub', 'shop_own_product', 'photo_app_pitch', 'dev_beta', 'discount_link')


def own_media(text, om):
    """Own newsletter/podcast/blog/channel: 'my free ... newsletter/guide/ebook', or 'my newsletter/podcast/blog/YouTube
    channel' together with an invitation or an external (non-Reddit) link."""
    external = text.str.extract_all(om['url']).list.eval(~pl.element().str.contains(om['own_site_url'])).list.any().fill_null(False)
    return text.str.contains(om['free_offer']) | (text.str.contains(om['media']) & (text.str.contains(om['invite']) | external))


def self_promotion(text, sp):
    """Own product/app/book/site: a builder phrase ('I built/made/wrote a ... app/tool/guide/book') or an owner/maker role
    ('I'm the founder/owner/dev ...') together with an external link or an invitation; an indefinite maker role ('I'm ...
    an author/a developer') needs an external link. Reddit/imgur links do not count and negated roles ('I'm not the
    author') are dropped."""
    external = text.str.extract_all(sp['url']).list.eval(~pl.element().str.contains(sp['own_site_url'])).list.any().fill_null(False)
    invite = text.str.contains(sp['invite'])
    loose_role = text.str.extract_all(sp['role_any']).list.eval(~pl.element().str.contains(sp['negation'])).list.any().fill_null(False)
    return (((text.str.contains(sp['build']) | text.str.contains(sp['role_owner']) | text.str.contains(sp['role_maker']))
             & (external | invite)) | (loose_role & external))


def own_site(text, os):
    """Own website: '<site/blog/app/...> is mine', or 'my (own) site/website' together with an external (non-Reddit) link;
    'my site' followed by an injection-site word ('my site reaction', 'my site was red') is dropped."""
    external = text.str.extract_all(os['url']).list.eval(~pl.element().str.contains(os['own_site_url'])).list.any().fill_null(False)
    site = text.str.extract_all(os['my_site']).list.eval(~pl.element().str.contains(os['injection_exclude'])).list.any().fill_null(False)
    return text.str.contains(os['mine']) | (site & external)


def profile_pointer(text, pp):
    """A link/code/referral pointed to the author's bio or profile ('link's in my bio', 'I have a $50 code in my bio'); a
    match naming a stored account code ('the code stored in my profile') is dropped."""
    return text.str.extract_all(pp['pointer']).list.eval(~pl.element().str.contains(pp['exclude'])).list.any().fill_null(False)


def app_developer(text, ad):
    """Developer voice for an own app/tool: an indie/solo/new/iOS/Android/app/mobile developer role ('I'm an independent
    developer', 'I'm the iOS developer for ...'; negated roles dropped) or 'I('ve been) working on / putting together a(n)
    ... app/tracker/tool', each together with an external (non-Reddit) link or an invitation; an app-distribution call to
    action (TestFlight, beta testers/testing, 'try/download it here', 'available on the App Store/Google Play') with an
    external link; or a TestFlight/beta-testers mention with an invitation. Profession roles ('a software developer'),
    'get it (for) free' and a singular 'a beta tester' are not matched."""
    external = text.str.extract_all(ad['url']).list.eval(~pl.element().str.contains(ad['own_site_url'])).list.any().fill_null(False)
    invite = text.str.contains(ad['invite'])
    role = text.str.extract_all(ad['role']).list.eval(~pl.element().str.contains(ad['negation'])).list.any().fill_null(False)
    return dict(dev_role=role & (external | invite), dev_working_on=text.str.contains(ad['working_on']) & (external | invite),
                app_cta_link=text.str.contains(ad['cta']) & external, app_beta=text.str.contains(ad['beta']) & invite)


def creator_shop(text, cs):
    """Creator-store selling: a link to a link-in-bio or creator-store host (linktr.ee, stan.store, payhip, gumroad, ko-fi,
    ...), or an Etsy listing/shop link together with an own-product claim ('on my Etsy', 'I sell', 'I put together a ...
    guide/plan/program'); a negated claim ('this is not my Etsy store') is dropped."""
    own = text.str.extract_all(cs['own_product']).list.eval(~pl.element().str.contains(cs['negation'])).list.any().fill_null(False)
    return dict(creator_hub=text.str.contains(cs['hub']), shop_own_product=text.str.contains(cs['shop']) & own)


def dev_beta(text, db):
    """Developer voice without an app noun nearby: 'try/join/test the beta' (a match ending in 'beta blocker(s)' is
    dropped) or 'why I built this/it'."""
    return text.str.extract_all(db['pattern']).list.eval(~pl.element().str.contains(db['exclude'])).list.any().fill_null(False)


def match_copy(text, pairs):
    """Matching-only copy for the code/referral parts: runs of spaces collapsed and glued redemption words split
    ('Usecode' -> 'Use code', 'tosave50' -> 'to save 50'); the stored and exported text is never changed."""
    for pattern, replacement in pairs:
        text = text.str.replace_all(pattern, replacement)
    return text


def discount_link(text, norm, dl):
    """Explicit discount-link invitation ('my / here's a link ... discount or $N off', 'link to/for a discount', "here's $N
    off") together with an external (non-Reddit) link; a match whose last word names a savings card or program is dropped.
    Share-link paths alone (e.g. /s/...) and prices alone never count."""
    external = text.str.extract_all(dl['url']).list.eval(~pl.element().str.contains(dl['own_site_url'])).list.any().fill_null(False)
    phrase = norm.str.extract_all(dl['phrase']).list.eval(~pl.element().str.contains(dl['exclude'])).list.any().fill_null(False)
    return phrase & external


def promotion_parts(frame, rule):
    """One flag per solicitation structure (structure only, no brand or vendor list); `discount_offer` counts only with a
    solicitation handle (own code/link, message-me, vendor voice, a code token), `giveaway`, `app_builder` and
    `service_offer` (our service + first-period/discount benefit + booking/contact action) need every listed pattern.
    `code_offer` matches 'code: TOKEN' / 'use code TOKEN' together with the word before it, and a match whose preceding
    word names a non-promotional code (diagnosis, billing, bank, approval, tracking, ...) is dropped. The code/referral
    parts read the matching copy (`match_copy`); URLs are always taken from the original text."""
    text = pl.col('text')
    norm = match_copy(text, rule['match_normalize'])
    urls = text.str.extract_all(r'https?://\S+').list.eval(
        pl.element().str.contains(rule['referral_url']) & ~pl.element().str.contains(rule['url_exclude']))
    return frame.select(
        'rowid',
        code_or_referral=norm.str.contains(rule['code_or_referral']),
        referral_url=urls.list.any().fill_null(False),
        provider_listing=text.str.contains(rule['provider_listing']),
        solicitation=text.str.contains(rule['solicitation']),
        promo_code_offer=norm.str.contains(rule['promo_code_offer']),
        giveaway=pl.all_horizontal([text.str.contains(p) for p in rule['giveaway']]),
        discount_offer=norm.str.contains(rule['discount_offer']) & norm.str.contains(rule['offer_handle']),
        code_offer=norm.str.extract_all(rule['code_offer']).list.eval(~pl.element().str.contains(rule['code_exclude']))
        .list.any().fill_null(False),
        app_builder=pl.all_horizontal([text.str.contains(p) for p in rule['app_builder']]),
        self_promo=self_promotion(text, rule['self_promo']),
        prize_draw=text.str.contains(rule['prize_draw']),
        service_offer=pl.all_horizontal([text.str.contains(p) for p in rule['service_offer']]),
        link_bio=text.str.contains(rule['link_bio']),
        promo_code=pl.any_horizontal([norm.str.contains(p) for p in rule['promo_code']]),
        own_media=own_media(text, rule['own_media']),
        care_team_text=text.str.contains(rule['care_team_text']),
        own_site=own_site(text, rule['own_site']),
        own_disclosure=text.str.contains(rule['own_disclosure']),
        obfuscated_code=text.str.contains(rule['obfuscated_code']),
        glued_code=text.str.contains(rule['glued_code']),
        consult_free=text.str.contains(rule['consult_free']),
        referral_offer=norm.str.contains(rule['referral_offer']),
        profile_pointer=profile_pointer(text, rule['profile_pointer']),
        **app_developer(text, rule['app_developer']),
        **creator_shop(text, rule['creator_shop']),
        photo_app_pitch=text.str.contains(rule['photo_app_pitch']['photo']) & text.str.contains(rule['photo_app_pitch']['pitch']),
        dev_beta=dev_beta(text, rule['dev_beta']),
        discount_link=discount_link(text, norm, rule['discount_link']))


def promotion(frame, rule):
    return promotion_parts(frame, rule).select('rowid', promo=pl.any_horizontal(PROMO_PARTS), *PROMO_PARTS)


# Common Spanish/Portuguese/French/German function words (English function words removed): a language signal, not a topic list.
FOREIGN_STOP = sorted(set(
    'de la que el en y los se del las por un para con una su al lo como más pero sus le ya este porque esta entre cuando muy sin '
    'sobre también hasta hay donde desde todo nos durante todos uno les ni contra otros ese eso ante ellos esto mí antes algunos '
    'qué unos yo otro otras otra él tanto esa estos mucho nada muchos cual poco ella estar estas algunas algo nosotros mi tengo '
    'estoy semanas bajado não uma os um com mais mas foi ao ele das tem à seu sua ou ser quando muito há já está eu só pelo pela '
    'até isso depois sem mesmo aos ter seus quem nas esse eles estão você tinha foram essa num nem suas meu às minha têm numa '
    'pelos elas havia seja será nós lhe deles essas esses pelas fosse dele le les des est pas une dans qui pour sur avec plus ce '
    'il ne mais sont du nous vous je ont été leur tout aussi bien peut fait ces cette mon même sans deux sous très der die und den '
    'das ist nicht ich zu mit sich des auf für ein eine dem es von wie auch als an nach bei einen um aus noch nur wird habe sind war '
    'ab mir mich'.split()) - set(FUNCTION.split('|')))


def language_counts(frame):
    """English function words vs common foreign function words per record (for strict_non_english)."""
    toks = pl.col('text').str.to_lowercase().str.extract_all(r"[a-zà-ÿ']+")
    fn = sorted(set(FUNCTION.split('|')))
    return frame.select('rowid', lang_en=toks.list.eval(pl.element().is_in(fn)).list.sum().fill_null(0),
                        lang_fo=toks.list.eval(pl.element().is_in(FOREIGN_STOP)).list.sum().fill_null(0))


def flags(a):
    c = pl.col
    make = {
        'strict_promotion': lambda r: c('promo').fill_null(False),
        'strict_non_english': lambda r: (c('lang_fo') >= r['min_foreign_words']) & (c('lang_fo') >= r['min_ratio_vs_english'] * c('lang_en')),
        'strict_repetitive': lambda r: (c('content_n') >= r['min_content_words']) & (c('top_share') > r['max_top_share']),
        'strict_mostly_borrowed': lambda r: (c('content_n') >= r['min_content_words'])
        & (c('original_n') < r['max_original_share'] * c('content_n')),
        'strict_info_below': lambda r: c('original_types') < r['min_original_types'],
    }
    return {name: make[name](rule) for name, rule in a.items()}


def run(base_dir, version_dir, output_dir, policy, work=None):
    """Write the strict-subset decision overlay (one parquet per shard) and a manifest; never touches inputs."""
    rules = validate_policy(policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    version_manifest = version_dir / 'manifest.json'
    work = work or output_dir / 'work'
    spec = dict(policy=policy, code_sha256=code_sha, version_dir=str(version_dir), features_dir=str(work),
                version_manifest_sha256=hashlib.sha256(version_manifest.read_bytes()).hexdigest())
    spec_path = output_dir / 'run-spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise SystemExit(f'{output_dir} was started with a different policy/code/version; use a new output directory')
    spec_path.write_text(json.dumps(spec, indent=1) + '\n')
    started = time.perf_counter()
    feat_seconds = run_features(base_dir, version_dir, work)
    marks = flags(rules)
    reason = pl.lit(None, dtype=pl.String)
    for name in reversed([r for r in REASONS if r in marks]):
        reason = pl.when(marks[name]).then(pl.lit(name)).otherwise(reason)
    (output_dir / 'decisions').mkdir(exist_ok=True)
    totals = dict(candidates=0, characters=0, removed=0, removed_characters=0)
    by_reason, standalone = {}, {n: 0 for n in marks}
    promo_parts = {n: 0 for n in PROMO_PARTS} if 'strict_promotion' in rules else None
    for shard in sorted((version_dir / 'decisions').glob('part-*.parquet')):
        prev = pl.read_parquet(shard)
        feats = pl.read_parquet(work / 'features' / shard.name)
        if 'strict_promotion' in rules or 'strict_non_english' in rules:
            text = (pl.read_parquet(base_dir / 'candidates' / shard.name, columns=['rowid', 'text'])
                    .join(feats.select('rowid'), on='rowid', how='semi'))
        if 'strict_non_english' in rules:
            feats = feats.join(language_counts(text), on='rowid', how='left')
        if 'strict_promotion' in rules:
            feats = feats.join(promotion(text, rules['strict_promotion']), on='rowid', how='left')
            for n in PROMO_PARTS:
                promo_parts[n] += int(feats[n].fill_null(False).sum())
        feats = feats.with_columns(reason_new=reason, **{f'hit_{n}': e for n, e in marks.items()})
        dec = (prev.join(feats.select('rowid', 'reason_new', *[f'hit_{n}' for n in marks]), on='rowid', how='left')
               .with_columns(version_decision=pl.col('decision'),
                             decision=pl.when(pl.col('reason_new').is_not_null()).then(pl.lit('exclude')).otherwise(pl.col('decision')),
                             reason=pl.when(pl.col('reason_new').is_not_null()).then(pl.col('reason_new')).otherwise(pl.col('reason')))
               .drop('reason_new'))
        dec.write_parquet(output_dir / 'decisions' / shard.name, compression='zstd')
        now = dec.filter(pl.col('version_decision') == 'candidate')
        keep = now.filter(pl.col('decision') == 'candidate')
        gone = now.filter(pl.col('decision') == 'exclude')
        totals['candidates'] += keep.height
        totals['characters'] += int(keep['characters'].sum() or 0)
        totals['removed'] += gone.height
        totals['removed_characters'] += int(gone['characters'].sum() or 0)
        for r, n, ch in gone.group_by('reason').agg(pl.len(), pl.col('characters').sum()).iter_rows():
            by_reason.setdefault(r, dict(records=0, characters=0))
            by_reason[r]['records'] += n
            by_reason[r]['characters'] += int(ch)
        for n in marks:
            standalone[n] += int(now[f'hit_{n}'].fill_null(False).sum())
    manifest = dict(revision=policy.get('revision'), layer='strict subset (lossy selection, not denoising)',
                    version_dir=str(version_dir), features_dir=str(work), extra_thresholds=rules, precedence=[r for r in REASONS if r in marks],
                    counts=totals, removed_by_reason=by_reason, standalone_hits=standalone, promotion_part_hits=promo_parts,
                    metrics=dict(seconds=round(time.perf_counter() - started, 1), feature_seconds=round(feat_seconds, 1),
                                 peak_rss_mib=round(peak_mib())))
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=Path('.cache/reddit-cleaning/denoise-x10-2026-09-25'))
    parser.add_argument('--version-dir', type=Path, default=Path('.cache/reddit-cleaning/denoise-x20-a2-2026-09-26'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--features-dir', type=Path, default=Path('.cache/reddit-cleaning/strict-features-a2-2026-09-26'))
    parser.add_argument('--policy', type=Path, default=Path('configs/reddit-strict-subset.json'))
    parser.add_argument('--features-only', action='store_true')
    args = parser.parse_args()
    if args.features_only:
        seconds = run_features(args.base_dir, args.version_dir, args.features_dir)
        return print(json.dumps(dict(feature_seconds=round(seconds, 1), peak_rss_mib=round(peak_mib()))))
    if args.output_dir is None:
        parser.error('--output-dir is required unless --features-only')
    print(json.dumps(run(args.base_dir, args.version_dir, args.output_dir, json.loads(args.policy.read_text()), args.features_dir),
                     indent=1))


if __name__ == '__main__':
    main()
