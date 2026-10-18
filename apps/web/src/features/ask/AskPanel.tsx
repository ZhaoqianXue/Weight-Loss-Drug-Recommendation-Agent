import { ArrowRight, ArrowUp, Check, CircleAlert, Loader2, X } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"

import { streamAsk } from "@/api/client"
import { useAskStatus } from "@/api/queries"
import { BrandMark } from "@/components/brand-mark"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { Explorer } from "@/state"

import { AnswerCard } from "./AnswerCard"
import { applyEvent, type Step, type Turn } from "./progress"

/** Questions from the agent's evaluation set, one per kind of question it handles. */
const EXAMPLES = [
  { kind: "Population", question: "Do women and men report different side effects on tirzepatide?" },
  { kind: "Compared with FAERS", question: "How does vomiting on Trulicity compare with what FAERS reports show?" },
  { kind: "Consequences", question: "Which side effects do writers most often say led them to stop Wegovy?" },
  { kind: "In their words", question: "How do people describe constipation on Zepbound in their own words?" },
  { kind: "Medical advice", question: "Should I stop taking Zepbound if I keep throwing up?" },
]

function StepRow({ step }: { step: Step }) {
  return (
    <li className="grid grid-cols-[16px_1fr] gap-x-2 gap-y-0.5">
      <span className="mt-[3px] flex size-4 items-center justify-center">
        {step.status === "running" ? (
          <Loader2 className="size-3.5 animate-spin text-ink" />
        ) : step.status === "error" ? (
          <X className="size-3.5 text-muted-foreground" />
        ) : (
          <Check className="size-3.5 text-[#2f6b45]" />
        )}
      </span>
      <span className={cn("text-[12px] text-prose", step.status === "error" && "text-muted-foreground")}>
        {step.title}
        {step.note && step.status === "done" && <span className="text-muted-foreground"> · {step.note}</span>}
      </span>
      {(step.details.length > 0 || (step.status === "error" && step.note)) && (
        <span className="col-start-2 flex flex-wrap gap-1">
          {step.details.map((detail) => (
            <span key={detail} className="rounded-full border border-hairline bg-card px-1.5 text-[10.5px] text-prose">
              {detail}
            </span>
          ))}
          {step.status === "error" && step.note && <span className="text-[10.5px] text-muted-foreground">Rejected: {step.note}</span>}
        </span>
      )}
    </li>
  )
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])
  return <span className="font-mono text-[11px] font-normal text-muted-foreground">{Math.max(0, Math.round((now - since) / 1000))} s</span>
}

function TurnView({ turn, onShowInGraph }: { turn: Turn; onShowInGraph: (update: Partial<Explorer>) => void }) {
  return (
    <div className="grid scroll-mt-3 gap-2.5" data-turn={turn.id}>
      <div className="ml-auto w-fit max-w-[80%] rounded-[13px_13px_3px_13px] bg-primary px-3 py-2.5 text-[#eaf0f7] dark:text-primary-foreground">{turn.question}</div>
      <div>
        <div className="min-w-0 rounded-[3px_13px_13px_13px] border border-tint-border bg-tint px-3.5 py-3 text-prose">
          {turn.status === "done" && turn.result ? (
            <AnswerCard result={turn.result} onShowInGraph={onShowInGraph} />
          ) : (
            <div className="grid gap-2">
              <p className="flex items-baseline justify-between gap-3 text-[13px] font-medium text-ink">
                {turn.status === "error" ? "No answer" : "Querying the knowledge graph"}
                {turn.status === "running" && <Elapsed since={turn.startedAt} />}
              </p>
              {turn.steps.length > 0 && <ol className="grid gap-1.5">{turn.steps.map((step) => <StepRow key={step.key} step={step} />)}</ol>}
              {turn.status === "error" && (
                <p className="flex items-start gap-1.5 text-[12px] text-destructive">
                  <CircleAlert className="mt-0.5 size-3.5 shrink-0" />
                  {turn.error}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

type Props = {
  /** A question sent from elsewhere on the page (e.g. the graph); `id` changes with every request. */
  request: { id: number; question: string } | null
  /** Set the graph explorer to what an answer counted. */
  onShowInGraph: (update: Partial<Explorer>) => void
}

export function AskPanel({ request, onShowInGraph }: Props) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState("")
  const status = useAskStatus()
  const nextId = useRef(1)
  const scroller = useRef<HTMLDivElement>(null)
  const running = turns.some((turn) => turn.status === "running")
  const available = status.data?.available ?? false

  const ask = useCallback(async (question: string) => {
    const id = nextId.current++
    setTurns((current) => [...current, { id, question, status: "running", steps: [], startedAt: Date.now() }])
    const update = (fn: (turn: Turn) => Turn) => setTurns((current) => current.map((turn) => (turn.id === id ? fn(turn) : turn)))
    try {
      await streamAsk(question, (event) => update((turn) => applyEvent(turn, event)))
      update((turn) => (turn.status === "running" ? { ...turn, status: "error", error: "The answer stream ended early." } : turn))
    } catch (error) {
      update((turn) => ({ ...turn, status: "error", error: error instanceof Error ? error.message : String(error) }))
    }
  }, [])

  useEffect(() => {
    if (request) void ask(request.question)
  }, [request, ask])

  // Bring a new question into view, and again when its answer arrives, so the answer is read from its start.
  const last = turns.at(-1)
  useEffect(() => {
    if (!last) return
    scroller.current?.querySelector<HTMLElement>(`[data-turn="${last.id}"]`)?.scrollIntoView({ behavior: "smooth", block: "start" })
  }, [last?.id, last?.status]) // eslint-disable-line react-hooks/exhaustive-deps -- only on a new turn or a finished one

  function submit(event: React.FormEvent) {
    event.preventDefault()
    const question = input.trim()
    if (!question || running || !available) return
    setInput("")
    void ask(question)
  }

  return (
    <div className="flex h-full flex-col bg-card">
      <div className="flex shrink-0 items-center gap-2.5 border-b border-hairline px-4 pt-[13px] pb-3">
        <BrandMark className="size-7" />
        <div className="grid min-w-0 gap-0.5">
          <h2 className="font-display text-[19px] leading-[1.2] font-normal text-ink">AskGLP1 agent</h2>
          <p className="text-meta text-muted-foreground">Answers from the knowledge graph. Every number links to the row it came from.</p>
        </div>
      </div>
      <div ref={scroller} className="relative flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto px-4 py-4" role="log" aria-live="polite" aria-label="Questions and answers">
        {turns.length === 0 ? (
          <div className="my-auto grid gap-7 py-2">
            <div className="grid justify-items-center gap-2 text-center">
              <h3 className="font-display text-[22px] leading-[1.2] font-normal text-ink">What would you like to know?</h3>
              <p className="max-w-[46ch] text-prose">
                Ask what Reddit writers and WebMD reviewers report about GLP-1 receptor agonists and how it compares with FAERS. Each
                question is answered on its own, from the graph only, without medical advice.
              </p>
            </div>
            <div className="grid gap-1.5">
              <p className="label-caps">Try asking</p>
              {EXAMPLES.map((example) => (
                <button
                  key={example.question}
                  type="button"
                  disabled={!available}
                  onClick={() => void ask(example.question)}
                  className="grid grid-cols-[1fr_auto] items-center gap-x-3 rounded-[10px] border border-chip-border bg-card px-3 py-[9px] text-left transition-colors hover:bg-tint disabled:opacity-50"
                >
                  <span className="grid gap-px">
                    <span className="text-[10px] tracking-[0.06em] text-subtle uppercase">{example.kind}</span>
                    <span className="text-ink">{example.question}</span>
                  </span>
                  <ArrowRight className="size-3.5 text-muted-foreground" aria-hidden />
                </button>
              ))}
            </div>
          </div>
        ) : (
          turns.map((turn) => <TurnView key={turn.id} turn={turn} onShowInGraph={onShowInGraph} />)
        )}
      </div>
      <form onSubmit={submit} className="grid shrink-0 gap-1.5 border-t border-hairline px-3.5 py-3">
        <div className="flex items-center gap-2 rounded-[14px] border border-input bg-card py-1.5 pr-1.5 pl-3 transition-colors focus-within:border-ring hover:border-ring">
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={available ? "Ask a question…" : (status.data?.reason ?? "Checking the agent…")}
            aria-label="Question for AskGLP1"
            maxLength={500}
            disabled={!available}
            className="min-w-0 flex-1 bg-transparent py-1 text-base outline-none placeholder:text-muted-foreground disabled:opacity-60 sm:text-ui"
          />
          <Button type="submit" size="icon-sm" aria-label="Send" disabled={!input.trim() || running || !available} className="size-8 rounded-[10px]">
            <ArrowUp />
          </Button>
        </div>

      </form>
    </div>
  )
}
