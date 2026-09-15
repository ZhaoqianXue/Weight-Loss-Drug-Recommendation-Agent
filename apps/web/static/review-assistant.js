/* Dataset-backed dialogue. No model calls or external medical knowledge. */
(function(root) {
    const normalize = value => String(value || '').trim().toLowerCase();
    const action = (label, type, data = {}) => {
        const medications = data.brands?.length > 2 ? 'these medications' : data.brands?.join(' and ');
        const questions = {
            pick: 'What do reviewers say about these medications?',
            'compare-pick': data.first ? `How does ${data.first} compare with another medication?` : 'How do patient experiences compare between medications?',
            'effect-pick': medications ? `What side effects do reviewers report with ${medications}?` : 'What side effects do people report?',
            choose: data.first ? `How does ${data.first} compare with ${data.brand}?` : data.purpose === 'compare-pick' ? `Let’s start with ${data.brand}.` : `What do reviewers say about ${data.brand}?`,
            overview: data.effect ? `How often do reviews mention ${data.effect}${medications ? ' with ' + medications : ''}?` : label === 'Back to results' ? 'Can you show me those results again?' : `How do ${medications} compare on ${data.mode === 'ratings' ? 'reviewer ratings' : data.mode === 'effects' ? 'reported side effects' : 'ratings and reported side effects'}?`,
            reviews: data.offset ? 'Could I see a few more reviews?' : `Can I read some reviews${data.effect ? ' mentioning ' + data.effect : ''}${medications ? ' for ' + medications : ''}?`,
            graph: `Can you show ${medications}${data.effect ? ' and the reports of ' + data.effect : ''} in the graph?`,
            'reset-graph': 'Can you show all medications in the graph again?',
            home: 'Let’s start a new question.',
            explain: ({source:'Where do these answers come from?', lines:'Does a thicker line mean a higher risk?', sample:'Why do some medications have fewer connections?', cause:'Does the graph prove that a medication caused a symptom?', limits:'What can’t this review data tell me?'})[data.topic] || 'How should I interpret this evidence?'
        };
        return { label, type, ...data, message: questions[type] || `Can you help me with ${label.toLowerCase()}?` };
    };
    class ReviewAssistant {
        constructor(rawRows) {
            this.rows = rawRows.map((r, i) => {
                let info = null;
                try { info = JSON.parse(r.standardized_info); } catch (_) { /* Missing extraction stays unknown. */ }
                if (['pending','failed'].includes(r['Extraction Status']) || r['Standardization Status'] === 'pending') info = null;
                if (!info || typeof info !== 'object' || !Array.isArray(info.side_effects)) info = null;
                const effects = new Set();
                for (const item of info?.side_effects || []) {
                    if (!item || typeof item !== 'object') continue;
                    for (const name of Array.isArray(item.name) ? item.name : [item.name]) {
                        if (typeof name === 'string' && name.trim()) effects.add(normalize(name));
                    }
                }
                return { raw: r, id: `R${String(i + 1).padStart(4, '0')}`, brand: r['Brand Name'], effects, valid: !!info };
            }).filter(r => r.brand);
            this.brands = [...new Set(this.rows.map(r => r.brand))].sort();
            this.effects = [...new Set(this.rows.flatMap(r => [...r.effects]))].sort((a,b) => b.length - a.length);
            this.context = { brands: [], effect: null };
            this.pending = null;
        }
        menu() {
            this.pending = null;
            return { title: 'Explore medication experiences', paragraphs: [
                `I can help you explore ${this.rows.length.toLocaleString('en-US')} WebMD reviews of ${this.brands.length} medications. What would you like to know?`
            ], actions: [action('Explore a medication','pick'), action('Compare patient experiences','compare-pick'), action('Explore reported side effects','effect-pick',{brands:[]}), action('Understand the evidence','explain')] };
        }
        stats(brand) {
            const rows = this.rows.filter(r => r.brand === brand);
            const counts = new Map();
            rows.forEach(r => r.effects.forEach(e => counts.set(e, (counts.get(e) || 0) + 1)));
            const ratings = {};
            for (const field of ['Effectiveness','Ease of Use','Satisfaction']) {
                const values = rows.map(r => String(r.raw[field] ?? '').trim()).filter(v => v !== '' && Number.isFinite(Number(v))).map(Number).filter(v => v >= 1 && v <= 5);
                ratings[field] = { n: values.length, mean: values.length ? values.reduce((a,b) => a+b,0)/values.length : null };
            }
            return { brand, n: rows.length, valid: rows.filter(r=>r.valid).length, ratings, effects: [...counts].sort((a,b)=>b[1]-a[1] || a[0].localeCompare(b[0])) };
        }
        overview(brands, effect = null, mode = 'both') {
            this.pending = null;
            this.context = { brands, effect };
            const stats = brands.map(b=>this.stats(b));
            const table = [['Reviews', ...stats.map(s=>String(s.n))]];
            if (mode !== 'effects') for (const field of ['Effectiveness','Ease of Use','Satisfaction']) {
                table.push([`${field} / 5`, ...stats.map(s=>s.ratings[field].n ? `${s.ratings[field].mean.toFixed(2)} (n=${s.ratings[field].n})` : 'Not available')]);
            }
            const chosen = effect ? [effect] : [...new Set(stats.flatMap(s=>s.effects.slice(0,3).map(e=>e[0])))].slice(0,6);
            if (mode !== 'ratings') for (const name of chosen) {
                table.push([name, ...stats.map(s=>{const count=new Map(s.effects).get(name)||0; return s.valid ? `${count}/${s.n} (${(100*count/s.n).toFixed(1)}%)` : 'Not available';})]);
            }
            const paragraphs = [effect ? `Reviews mentioning “${effect}” in the extracted data.` : 'Here are the ratings and reports in this review dataset.'];
            if (stats.some(s=>s.valid<s.n)) paragraphs[0] += ` Side-effect extraction is incomplete: ${stats.map(s=>`${s.brand} ${s.valid}/${s.n}`).join('; ')} processed.`;
            if (!effect && mode !== 'ratings') paragraphs.push('Side-effect rows combine up to three most frequently extracted terms per medication, capped at six distinct terms.');
            if (mode !== 'ratings') paragraphs.push('Percentages are shares of collected reviews, not clinical incidence rates. A zero means no extracted match, not absence of the symptom.');
            if (brands.length > 1) paragraphs.push('These groups are self-selected and not matched. Differences do not establish comparative effectiveness or safety.');
            const small = stats.filter(s=>s.n<50);
            if (small.length) paragraphs.push(`Limited sample: ${small.map(s=>`${s.brand} has ${s.n} reviews`).join('; ')}. Interpret descriptive differences cautiously.`);
            if (stats.some(s=>s.valid<s.n)) paragraphs.push('Some reviews are not yet extracted; their side effects are unknown. Counts describe the processed subset and use all collected reviews as the denominator. Extraction coverage is shown below.');
            const actions = [action('See supporting reviews','reviews',{brands,effect}), action(brands.length>2?'Show medications in graph':brands.length>1?'Show both in graph':'Show in graph','graph',{brands,effect}), action('Explore a side effect','effect-pick',{brands})];
            if (brands.length===1) actions.push(action('Compare with another','compare-pick',{first:brands[0]}));
            else actions.push(action('Ratings and side effects','overview',{brands,mode:'both'}),action('Ratings only','overview',{brands,mode:'ratings'}),action('Side effects only','overview',{brands,mode:'effects'}));
            actions.push(action('Understand these numbers','explain'),action('Start over','home'));
            return { title: brands.length>2 ? (effect ? `Reported ${effect} across ${brands.length} medications` : `Review data for ${brands.length} medications`) : brands.join(' vs '), paragraphs, headers:['Measure',...brands], table,
                source:`Source: standardized_reviews_all.csv · extraction coverage ${stats.map(s=>`${s.brand} ${s.valid}/${s.n}`).join('; ')}. Side-effect terms are case-folded; medical synonyms are not automatically merged.`, actions };
        }
        reviews(brands, effect, offset=0) {
            this.pending = null;
            this.context = { brands, effect };
            const matches = this.rows.filter(r=>(!brands.length || brands.includes(r.brand)) && (!effect || r.effects.has(effect)));
            let available = matches.filter(r=>String(r.raw['Textual Review']||'').trim());
            if (brands.length > 1) {
                const groups = brands.map(b=>available.filter(r=>r.brand===b));
                available = [];
                for (let i=0; i<Math.max(0,...groups.map(g=>g.length)); i++) {
                    groups.forEach(group=>{if(group[i]) available.push(group[i]);});
                }
            }
            const items = available.slice(offset, offset+3).map(r=>({id:r.id,brand:r.brand,date:r.raw.Date,text:r.raw['Textual Review']}));
            const actions = [];
            if (offset+3<available.length) actions.push(action('Next 3 reviews','reviews',{brands,effect,offset:offset+3}));
            actions.push(action('Back to results','overview',{brands:brands.length?brands:this.brands,effect}),action('Start over','home'));
            return {title:'Source review evidence',paragraphs:[`${matches.length} matching records; ${available.length} contain review text. ${items.length ? `Showing ${offset+1}–${offset+items.length} ${brands.length>1?'interleaved by medication, preserving CSV order within each medication':'in CSV order'}.` : 'No matching review text is available.'}`,
                'These are original excerpts, not generated summaries or a representative sample. Matching uses extracted terms, so the original wording may differ; review the text to verify the extraction.'], reviews:items,
                source:'Review IDs identify data-row order in this CSV snapshot (header excluded). Expand a record to read its full text. Usernames are omitted.',actions};
        }
        explain(topic) {
            const topics = {
                source:['Where answers come from','WebMD reviews → structured extraction → standardized terms → deterministic calculations over the loaded CSV. This interactive assistant does not call an LLM or query Neo4j. Pending reviews contribute original text and ratings; their side effects remain unknown. The repository contains separate TableRAG/GraphRAG research components. Original review text is available through “See supporting reviews”.'],
                lines:['What graph lines mean','A line connects entities in extracted relations. Its weight counts relation occurrences, and its width scales with the square root of that weight. It is not a risk estimate. Chat percentages instead count each review once per normalized term, so they can differ from graph counts.'],
                sample:['Why fewer connections do not mean fewer risks',`The dataset contains ${this.stats('Zepbound').n} Zepbound reviews and ${this.stats('Mounjaro').n} Mounjaro reviews. Connections also depend on extraction and active filters. A smaller number of connections does not establish greater safety.`],
                cause:['Association is not proof of causation','Relations come from self-reported reviews and automated extraction. A relation called “causes” in the data records an extracted report; it does not establish causality. The dataset is not a controlled trial or a complete list of adverse effects.'],
                limits:['What this evidence cannot answer','These reviews cannot determine the best medication or dose for an individual, confirm current approval status, or estimate clinical incidence. Reporting is self-selected, sample sizes differ, and extraction can contain errors. No independent clinical or human answer-quality validation has been completed.']
            };
            if (topic && topics[topic]) return {title:topics[topic][0],paragraphs:[topics[topic][1]],actions:[action('Other evidence questions','explain'),action('Start over','home')]};
            return {title:'Understand the evidence',paragraphs:['What would you like to clarify?'],actions:Object.entries(topics).map(([key,v])=>action(v[0],'explain',{topic:key})).concat(action('Start over','home'))};
        }
        run(a) {
            if (a.type==='home') { this.context={brands:[],effect:null}; return this.menu(); }
            if (a.type==='pick' || a.type==='compare-pick') {
                this.pending={type:a.type,first:a.first};
                return {title:a.type==='pick'?'Which medication?':a.first?`Compare ${a.first} with…`:'Choose the first medication',paragraphs:[],actions:this.brands.filter(b=>b!==a.first).map(b=>action(b,'choose',{brand:b,purpose:a.type,first:a.first}))};
            }
            if (a.type==='choose') {
                if (a.purpose==='compare-pick' && !a.first) return this.run({type:'compare-pick',first:a.brand});
                return this.overview(a.first?[a.first,a.brand]:[a.brand]);
            }
            if (a.type==='effect-pick') {
                const brands=a.brands || this.context.brands;
                this.context={brands,effect:null}; this.pending={type:'effect'};
                const counts=new Map();
                this.rows.filter(r=>!brands.length||brands.includes(r.brand)).forEach(r=>r.effects.forEach(e=>counts.set(e,(counts.get(e)||0)+1)));
                return {title:'Which reported side effect?',paragraphs:['Choose a reported effect or type one below.'],actions:[...counts].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,8).map(([effect])=>action(effect,'overview',{brands:brands.length?brands:this.brands,effect,mode:'effects'})).concat(action('Start over','home'))};
            }
            if (a.type==='overview') return this.overview(a.brands,a.effect||null,a.mode||'both');
            if (a.type==='reviews') return this.reviews(a.brands,a.effect,a.offset||0);
            if (a.type==='explain') {this.pending=null;return this.explain(a.topic);}
            if (a.type==='graph') return {title:'Graph updated',paragraphs:[`Showing ${a.brands.join(', ')}${a.effect?` with “${a.effect}”`:' with connected conditions and the current top-side-effect percentage'}. Chat statistics still describe the full dataset.`],graph:a,actions:[action('Reset graph','reset-graph'),action('Back to results','overview',{brands:a.brands,effect:a.effect})]};
            if (a.type==='reset-graph') return {title:'Graph reset',paragraphs:['All medications restored; top side effects reset to 1%.'],graph:a,actions:[action('Start over','home')]};
            return this.menu();
        }
        ask(input) {
            const q=normalize(input);
            if (/^(hi|hello|help|start over|reset|你好|重新开始)[.!！。]?$/.test(q)) return this.run({type:'home'});
            if (/dose|dosage|prescrib|should i|safe for me|best drug|best medication|safest|safest.*|剂量|推荐.*我|适合我|最安全/.test(q)) return this.explain('limits');
            if (/\b(?:causal|causation|prove)\b|因果/.test(q)) return this.explain('cause');
            if (/\b(?:thicker|thick|line|lines)\b|连线|粗细/.test(q)) return this.explain('lines');
            if (/fewer.*(?:connection|node)|sample size|样本/.test(q)) return this.explain('sample');
            if (/source|where.*(?:data|answer)|evidence|来源|证据/.test(q) && !/review|评论/.test(q)) return this.explain('source');
            if (/reset.*graph|重置.*图/.test(q)) return this.run({type:'reset-graph'});
            if (/price|cost|insurance|approval|fda|trial|pregnan|费用|价格|批准|孕/.test(q)) return {title:'Outside the supported evidence',paragraphs:['This assistant does not have a verified pricing, regulatory, clinical-trial or pregnancy evidence source. I will not infer an answer from review reports. I can compare reviewer ratings, extracted side effects, and source reviews.'],actions:[action('Explore the available evidence','explain')]};
            const genericNames = [...new Set(this.rows.map(r=>r.raw['Drug Name']).filter(Boolean))];
            const mentionedGenerics = genericNames.filter(g=>new RegExp(`\\b${g.toLowerCase()}\\b`).test(q));
            const brands=this.brands.filter(b=>new RegExp(`\\b${b.toLowerCase()}\\b`).test(q) || this.rows.some(r=>r.brand===b && mentionedGenerics.includes(r.raw['Drug Name']))).sort((a,b)=>q.indexOf(a.toLowerCase())-q.indexOf(b.toLowerCase()));

            const query=q.replace(/\bdiarrhea\b/g,'diarrhoea').replace(/恶心/g,' nausea ').replace(/便秘/g,' constipation ').replace(/呕吐/g,' vomiting ').replace(/腹泻/g,' diarrhoea ').replace(/疲劳/g,' fatigue ').replace(/头痛/g,' headache ');
            const effect=this.effects.find(e=>e.length>2 && (` ${query.replace(/[^\p{L}\p{N} ]/gu,' ')} `).includes(` ${e} `));
            if (mentionedGenerics.length && brands.length) return this.overview(brands,effect, /rating|评分/.test(q)?'ratings':effect?'effects':'both');
            if (brands.length>2) return {title:'Choose a pair',paragraphs:['For a readable comparison, choose two medications.'],actions:[action('Compare two medications','compare-pick')]};
            if (this.pending && brands.length===1 && !effect && /^(ozempic|wegovy(?: hd)?|mounjaro|zepbound|rybelsus|victoza(?: [23]-pak)?|saxenda|trulicity|semaglutide|tirzepatide|liraglutide|dulaglutide)$/.test(q)) return this.run({type:'choose',brand:brands[0],purpose:this.pending.type,first:this.pending.first});
            const selected=brands.length?brands:this.context.brands;
            if (/review|评论/.test(q) && /show|read|support|see|看|原文/.test(q)) return this.reviews(selected,effect||this.context.effect);
            if (/graph|图谱/.test(q) && selected.length) return this.run({type:'graph',brands:selected,effect:effect||this.context.effect});
            if (/compare|\bvs\b|versus|对比|比较/.test(q)) return brands.length===2?this.overview(brands,effect):this.run({type:'compare-pick',first:brands[0]||selected[0]});
            if (brands.length && /pros and cons|review summary|优缺点|总结/.test(q)) return this.overview(brands);
            if (brands.length===1 && /^(what about|how about)/.test(q) && !effect) return this.overview(brands,this.context.effect);
            if (effect) return this.overview(selected.length?selected:this.brands,effect,'effects');
            if (/side effect|副作用/.test(q)) return selected.length?this.overview(selected,null,'effects'):this.run({type:'effect-pick',brands:[]});
            if (/rating|satisfaction|effectiveness|ease of use|评分|满意度/.test(q) && selected.length) return this.overview(selected,null,'ratings');
            if (brands.length && /^(what do reviewers say about |tell me about |explore |overview of |介绍一下)?(ozempic|wegovy(?: hd)?|mounjaro|zepbound|rybelsus|victoza(?: [23]-pak)?|saxenda|trulicity|semaglutide|tirzepatide|liraglutide|dulaglutide)[?.。！]?$/.test(q)) return this.overview(brands);
            return {title:'Let’s make the question specific',paragraphs:['I couldn’t match that question to the review data. Try “Compare Wegovy and Ozempic,” or choose a topic below.'],actions:[action('Explore a medication','pick'),action('Compare medications','compare-pick'),action('Choose a side effect','effect-pick'),action('Evidence and limitations','explain')]};
        }
    }
    if (typeof module !== 'undefined' && module.exports) module.exports=ReviewAssistant;
    else root.ReviewAssistant=ReviewAssistant;
})(typeof window !== 'undefined' ? window : this);
