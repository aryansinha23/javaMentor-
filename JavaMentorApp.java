import java.io.IOException;
import java.io.PrintStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Java Concept &amp; Code Mentor.
 *
 * <p>A terminal chatbot that answers Java questions through a fixed three-phase
 * pipeline: conceptual breakdown, algorithmic blueprint, then a complete
 * compilable implementation with a test harness. A second model pass audits
 * every draft against a rubric and forces a bounded repair loop.
 *
 * <p>Self-contained on purpose: no build tool and no third-party libraries.
 * Run with {@code java JavaMentorApp.java} on JDK 17 or newer.
 *
 * <p>Requires the environment variable {@code GEMINI_API_KEY}. Optionally
 * {@code GEMINI_MODEL} overrides the default model.
 */
public final class JavaMentorApp {

    private JavaMentorApp() {
    }

    public static void main(String[] args) {
        ApiConfig config;
        try {
            config = ApiConfig.fromEnvironment();
        } catch (IllegalStateException ex) {
            System.err.println("Configuration error: " + ex.getMessage());
            System.err.println("""
                    Setup:
                      1. Get a key: https://aistudio.google.com/apikey
                      2. export GEMINI_API_KEY="your_key"
                      3. rerun: java JavaMentorApp.java""");
            System.exit(2);
            return;
        }

        GeminiClient client = new GeminiClient(config);
        ChatSession session = new ChatSession(new MentorAgent(client), new CriticAgent(client));

        try (Scanner scanner = new Scanner(System.in)) {
            new ConsoleUI(scanner, System.out, session).run(config.modelName(), config.maskedKey());
        }
    }

    // ---------------------------------------------------------------- config

    /** Immutable runtime configuration resolved from the environment. */
    static final class ApiConfig {

        private static final String BASE_URL =
                "https://generativelanguage.googleapis.com/v1beta/models/";
        private static final String DEFAULT_MODEL = "gemini-2.5-flash";
        private static final int MAX_RETRIES = 4;

        private final String apiKey;
        private final String model;

        private ApiConfig(String apiKey, String model) {
            this.apiKey = apiKey;
            this.model = model;
        }

        static ApiConfig fromEnvironment() {
            String key = System.getenv("GEMINI_API_KEY");
            if (key == null || key.isBlank()) {
                throw new IllegalStateException("GEMINI_API_KEY is not set.");
            }
            String model = System.getenv().getOrDefault("GEMINI_MODEL", DEFAULT_MODEL);
            return new ApiConfig(key.trim(), model.trim());
        }

        String apiKey() {
            return apiKey;
        }

        int maxRetries() {
            return MAX_RETRIES;
        }

        String endpoint() {
            return BASE_URL + model + ":generateContent";
        }

        String modelName() {
            return model;
        }

        String maskedKey() {
            return apiKey.length() <= 4 ? "****" : "****" + apiKey.substring(apiKey.length() - 4);
        }
    }

    // ------------------------------------------------------------- exception

    /** Typed failure from the remote API carrying retryability and status context. */
    static final class MentorApiException extends RuntimeException {

        private final boolean retryable;

        private MentorApiException(String message, boolean retryable, Throwable cause) {
            super(message, cause);
            this.retryable = retryable;
        }

        boolean isRetryable() {
            return retryable;
        }

        static MentorApiException network(Throwable cause) {
            return new MentorApiException("Network failure contacting the model API.", true, cause);
        }

        static MentorApiException forStatus(int status, String body) {
            boolean retryable = status == 429 || status >= 500;
            String detail = body == null ? "" : body.length() > 300 ? body.substring(0, 300) : body;
            return new MentorApiException("Model API returned HTTP " + status + ". " + detail,
                    retryable, null);
        }

        static MentorApiException malformed(String detail) {
            return new MentorApiException("Malformed model response: " + detail, true, null);
        }
    }

    // ---------------------------------------------------------------- client

    /** REST transport with bounded exponential backoff and jitter. */
    static final class GeminiClient {

        private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(120);
        private static final int BASE_BACKOFF_MS = 1_000;
        private static final int MAX_BACKOFF_MS = 16_000;

        private final HttpClient http;
        private final ApiConfig config;

        GeminiClient(ApiConfig config) {
            this.config = config;
            this.http = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(15))
                    .followRedirects(HttpClient.Redirect.NORMAL)
                    .build();
        }

        String generate(String systemInstruction, String userPrompt) {
            String payload = buildPayload(systemInstruction, userPrompt);
            MentorApiException last = null;

            for (int attempt = 1; attempt <= config.maxRetries(); attempt++) {
                try {
                    return execute(payload);
                } catch (MentorApiException ex) {
                    last = ex;
                    if (!ex.isRetryable() || attempt == config.maxRetries()) {
                        break;
                    }
                    sleepQuietly(backoffMillis(attempt));
                }
            }
            throw last == null
                    ? new MentorApiException("Model call failed with no recorded cause.", false, null)
                    : last;
        }

        private String execute(String payload) {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(config.endpoint()))
                    .timeout(REQUEST_TIMEOUT)
                    .header("Content-Type", "application/json")
                    .header("x-goog-api-key", config.apiKey())
                    .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response;
            try {
                response = http.send(request,
                        HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            } catch (IOException ex) {
                throw MentorApiException.network(ex);
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                throw MentorApiException.network(ex);
            }

            if (response.statusCode() != 200) {
                throw MentorApiException.forStatus(response.statusCode(), response.body());
            }
            return extractText(response.body());
        }

        private String buildPayload(String systemInstruction, String userPrompt) {
            return "{\"system_instruction\":{\"parts\":[{\"text\":\""
                    + jsonEscape(systemInstruction)
                    + "\"}]},\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\""
                    + jsonEscape(userPrompt)
                    + "\"}]}],\"generationConfig\":{\"temperature\":0.35,\"topP\":0.95,"
                    + "\"maxOutputTokens\":8192}}";
        }

        private String extractText(String body) {
            int candidates = body.indexOf("\"candidates\"");
            if (candidates < 0) {
                String reason = valueOf(body, "\"blockReason\"");
                throw MentorApiException.malformed(
                        reason == null ? "no candidates returned" : "blocked: " + reason);
            }

            StringBuilder text = new StringBuilder();
            int cursor = candidates;
            while (true) {
                int key = body.indexOf("\"text\"", cursor);
                if (key < 0) {
                    break;
                }
                int colon = body.indexOf(':', key + 6);
                if (colon < 0) {
                    break;
                }
                int quote = body.indexOf('"', colon);
                if (quote < 0) {
                    break;
                }
                String part = readJsonString(body, quote);
                if (part == null) {
                    break;
                }
                text.append(part);
                cursor = quote + part.length() + 1;
            }

            String result = text.toString().trim();
            if (result.isEmpty()) {
                throw MentorApiException.malformed("empty completion text");
            }
            return result;
        }

        private String valueOf(String body, String key) {
            int at = body.indexOf(key);
            if (at < 0) {
                return null;
            }
            int colon = body.indexOf(':', at + key.length());
            int quote = colon < 0 ? -1 : body.indexOf('"', colon);
            return quote < 0 ? null : readJsonString(body, quote);
        }

        private static String readJsonString(String source, int quoteIndex) {
            StringBuilder sb = new StringBuilder();
            for (int i = quoteIndex + 1; i < source.length(); i++) {
                char c = source.charAt(i);
                if (c == '"') {
                    return sb.toString();
                }
                if (c != '\\') {
                    sb.append(c);
                    continue;
                }
                i++;
                if (i >= source.length()) {
                    return null;
                }
                char escape = source.charAt(i);
                switch (escape) {
                    case 'n' -> sb.append('\n');
                    case 't' -> sb.append('\t');
                    case 'r' -> sb.append('\r');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'u' -> {
                        if (i + 4 >= source.length()) {
                            return null;
                        }
                        sb.append((char) Integer.parseInt(source.substring(i + 1, i + 5), 16));
                        i += 4;
                    }
                    default -> sb.append(escape);
                }
            }
            return null;
        }

        private static String jsonEscape(String raw) {
            StringBuilder sb = new StringBuilder(raw.length() + 64);
            for (int i = 0; i < raw.length(); i++) {
                char c = raw.charAt(i);
                switch (c) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    case '\r' -> sb.append("\\r");
                    case '\t' -> sb.append("\\t");
                    default -> {
                        if (c < 0x20) {
                            sb.append(String.format("\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                    }
                }
            }
            return sb.toString();
        }

        private static int backoffMillis(int attempt) {
            int exponential = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS << (attempt - 1));
            return exponential + ThreadLocalRandom.current().nextInt(0, 500);
        }

        private static void sleepQuietly(int millis) {
            try {
                Thread.sleep(millis);
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }
    }

    // --------------------------------------------------------------- prompts

    /** The contract prompts that constrain model behaviour. */
    static final class SystemPrompts {

        private SystemPrompts() {
        }

        static final String MENTOR = """
                You are a senior Java platform engineer acting as a strict pedagogical mentor for
                university students studying Object-Oriented Programming with Java.

                Every answer MUST follow this exact three-phase pipeline, with these literal headings:

                [PHASE 1: CONCEPTUAL BREAKDOWN]
                - Theoretical grounding: define the concept and the internal JVM/runtime mechanism.
                - API and syntax mechanics: name the exact keywords, interfaces and java.* packages.
                - Industrial context: where this is used in production and which anti-pattern it replaces.

                [PHASE 2: ALGORITHMIC BLUEPRINT]
                - Numbered, ordered logic steps or pseudocode.
                - Edge cases: null-safety, concurrency hazards, boundary limits, exceptions to propagate.
                - Annotate Time Complexity and Space Complexity in Big-O where applicable.

                [PHASE 3: JAVA IMPLEMENTATION & EXECUTION]
                - One complete, compilable Java class. No placeholders, no "// logic goes here", no TODOs.
                - Oracle/Google style: 4-space indent, idiomatic names, Javadoc on public members.
                - A public static void main containing at least TWO distinct test scenarios with
                  descriptive input and the expected console output stated in comments.

                Rules:
                - Target Java 17. Use only the standard library.
                - If the question is not about Java, say so in one line, then still answer it in all
                  three phases using Java equivalents.
                - Never truncate code. Never summarise a method body.
                - Do not add greetings, sign-offs, or meta-commentary about being an AI.
                """;

        static final String CRITIC = """
                You are a code-review auditor for a Java tutoring chatbot. You receive a student
                question and a candidate answer. Grade the answer against this rubric only:

                1. All three phase headings present and each phase substantive.
                2. Phase 1 covers runtime mechanism, named API elements, and production context.
                3. Phase 2 has ordered logic, edge cases, and Big-O annotations.
                4. Phase 3 has a full compilable class with a main method and two test scenarios.
                5. No placeholder comments, no truncation, no syntax errors, no invented APIs.

                Reply with exactly one line starting with PASS or FAIL, followed by numbered defect
                descriptions when failing. Be terse. Do not rewrite the answer.
                """;

        static final String REPAIR_SUFFIX = """

                The draft above was rejected by the auditor with the defects listed below.
                Rewrite the COMPLETE answer from scratch, fixing every defect. Keep all three phases
                and keep the code whole. Return only the corrected answer.
                """;
    }

    // ---------------------------------------------------------------- agents

    /** Produces and repairs three-phase answers. */
    static final class MentorAgent {

        private final GeminiClient client;

        MentorAgent(GeminiClient client) {
            this.client = client;
        }

        String teach(String question) {
            return client.generate(SystemPrompts.MENTOR, question);
        }

        String repair(String question, String rejectedDraft, String defects) {
            String prompt = "Question:\n" + question
                    + "\n\nRejected draft:\n" + rejectedDraft
                    + "\n\nAuditor defects:\n" + defects
                    + SystemPrompts.REPAIR_SUFFIX;
            return client.generate(SystemPrompts.MENTOR, prompt);
        }
    }

    /** Second model pass that audits a draft against the delivery rubric. */
    static final class CriticAgent {

        record Critique(boolean passed, String feedback) {
            static Critique pass() {
                return new Critique(true, "");
            }
        }

        private final GeminiClient client;

        CriticAgent(GeminiClient client) {
            this.client = client;
        }

        Critique review(String question, String candidateAnswer) {
            String prompt = "Question:\n" + question + "\n\nCandidate answer:\n" + candidateAnswer;
            String verdict = client.generate(SystemPrompts.CRITIC, prompt).trim();

            if (verdict.regionMatches(true, 0, "PASS", 0, 4)) {
                return Critique.pass();
            }
            String defects = verdict.length() > 4 ? verdict.substring(4).trim() : verdict;
            return new Critique(false, defects.isBlank() ? "Unspecified rubric failure." : defects);
        }
    }

    // ------------------------------------------------------------ validation

    /** Outcome of the offline structural inspection of a draft. */
    record ValidationResult(boolean ok, List<String> issues) {

        ValidationResult {
            issues = List.copyOf(issues);
        }

        static ValidationResult pass() {
            return new ValidationResult(true, List.of());
        }

        static ValidationResult fail(List<String> issues) {
            return new ValidationResult(false, issues);
        }

        String renderedIssues() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < issues.size(); i++) {
                sb.append(i + 1).append(". ").append(issues.get(i)).append('\n');
            }
            return sb.toString();
        }
    }

    /**
     * Deterministic guard that runs before any model-based review.
     * Catches truncation and placeholder stubs without spending a second API call.
     */
    static final class ResponseValidator {

        private static final String[] PLACEHOLDER_MARKERS = {
                "logic goes here", "your code here", "implement me", "todo:", "// todo", "rest of code"
        };

        private ResponseValidator() {
        }

        static ValidationResult inspect(String draft) {
            List<String> issues = new ArrayList<>();
            String lower = draft.toLowerCase(Locale.ROOT);

            if (draft.length() < 400) {
                issues.add("Answer is too short to cover three substantive phases.");
            }
            require(draft, "[PHASE 1", "Missing [PHASE 1: CONCEPTUAL BREAKDOWN] section.", issues);
            require(draft, "[PHASE 2", "Missing [PHASE 2: ALGORITHMIC BLUEPRINT] section.", issues);
            require(draft, "[PHASE 3", "Missing [PHASE 3: JAVA IMPLEMENTATION & EXECUTION] section.", issues);
            require(lower, "public class", "No public class declaration in the implementation.", issues);
            require(lower, "public static void main", "No executable main method / test harness.", issues);

            for (String marker : PLACEHOLDER_MARKERS) {
                if (lower.contains(marker)) {
                    issues.add("Placeholder text detected: '" + marker + "'.");
                }
            }
            if (countOccurrences(draft, "...") > 3) {
                issues.add("Excessive ellipses suggest elided code.");
            }
            if (!lower.contains("complexity")) {
                issues.add("Missing algorithmic complexity annotation.");
            }
            return issues.isEmpty() ? ValidationResult.pass() : ValidationResult.fail(issues);
        }

        private static void require(String haystack, String needle, String message, List<String> issues) {
            if (!haystack.contains(needle)) {
                issues.add(message);
            }
        }

        private static int countOccurrences(String text, String token) {
            int count = 0;
            int index = text.indexOf(token);
            while (index >= 0) {
                count++;
                index = text.indexOf(token, index + token.length());
            }
            return count;
        }
    }

    // --------------------------------------------------------------- session

    /**
     * Orchestrates the pipeline: draft, inspect, audit, repair.
     * Repair rounds are capped so a stubborn model cannot loop forever.
     */
    static final class ChatSession {

        private static final int MAX_REPAIR_ROUNDS = 2;

        private final MentorAgent mentor;
        private final CriticAgent critic;

        ChatSession(MentorAgent mentor, CriticAgent critic) {
            this.mentor = mentor;
            this.critic = critic;
        }

        String answer(String question, RepairListener onRepair) {
            String draft = mentor.teach(question);

            ValidationResult structural = ResponseValidator.inspect(draft);
            CriticAgent.Critique verdict = structural.ok()
                    ? critic.review(question, draft)
                    : new CriticAgent.Critique(false, structural.renderedIssues());

            int round = 1;
            while (!verdict.passed() && round <= MAX_REPAIR_ROUNDS) {
                onRepair.onRepair(round, verdict.feedback());
                draft = mentor.repair(question, draft, verdict.feedback());

                ValidationResult recheck = ResponseValidator.inspect(draft);
                verdict = recheck.ok()
                        ? critic.review(question, draft)
                        : new CriticAgent.Critique(false, recheck.renderedIssues());
                round++;
            }
            return draft;
        }

        @FunctionalInterface
        interface RepairListener {
            void onRepair(int round, String feedback);
        }
    }

    // --------------------------------------------------------------- utility

    /** Normalises raw console input before it reaches the model. */
    static final class InputSanitizer {

        static final int MAX_LENGTH = 2_000;
        private static final Set<String> EXIT_TOKENS = Set.of("exit", "quit", ":q", "bye");

        private InputSanitizer() {
        }

        static String sanitize(String raw) {
            if (raw == null) {
                return "";
            }
            String cleaned = raw.replaceAll("[\\p{Cntrl}&&[^\\n\\t]]", "")
                    .replaceAll("[\\t ]+", " ")
                    .trim();
            return cleaned.length() > MAX_LENGTH ? cleaned.substring(0, MAX_LENGTH) : cleaned;
        }

        static boolean isExitToken(String sanitized) {
            return EXIT_TOKENS.contains(sanitized.toLowerCase(Locale.ROOT));
        }

        static boolean isCommand(String sanitized) {
            return sanitized.startsWith("/");
        }
    }

    // -------------------------------------------------------------------- ui

    /** Interactive terminal front-end with command handling and graceful exit. */
    static final class ConsoleUI {

        private final Scanner scanner;
        private final PrintStream out;
        private final ChatSession session;

        ConsoleUI(Scanner scanner, PrintStream out, ChatSession session) {
            this.scanner = scanner;
            this.out = out;
            this.session = session;
        }

        void run(String modelName, String maskedKey) {
            banner(modelName, maskedKey);
            while (true) {
                out.print("\n> ");
                String raw = scanner.hasNextLine() ? scanner.nextLine() : null;
                if (raw == null) {
                    out.println("\nInput stream closed. Goodbye.");
                    return;
                }

                String input = InputSanitizer.sanitize(raw);
                if (input.isEmpty()) {
                    continue;
                }
                if (InputSanitizer.isExitToken(input)) {
                    out.println("Session closed. Keep building.");
                    return;
                }
                if (InputSanitizer.isCommand(input)) {
                    if (handleCommand(input)) {
                        return;
                    }
                    continue;
                }
                handleQuestion(input);
            }
        }

        private void handleQuestion(String question) {
            try {
                String answer = session.answer(question, this::printRepair);
                out.println("\n" + answer);
            } catch (RuntimeException ex) {
                out.println("\n[!] Could not complete the request: " + ex.getMessage());
                out.println("    The session is still alive - try rephrasing or type /help.");
            }
        }

        private boolean handleCommand(String command) {
            switch (command.toLowerCase(Locale.ROOT)) {
                case "/exit", "/quit" -> {
                    out.println("Session closed. Keep building.");
                    return true;
                }
                case "/help" -> out.println("""
                        Commands:
                          /help   show this list
                          /exit   end the session
                          exit    end the session
                        Ask any Java question to receive the three-phase answer.""");
                default -> out.println("Unknown command: " + command + ". Try /help.");
            }
            return false;
        }

        private void printRepair(int round, String feedback) {
            String trimmed = feedback.strip();
            int newline = trimmed.indexOf('\n');
            String line = newline > 0 ? trimmed.substring(0, newline) : trimmed;
            if (line.length() > 120) {
                line = line.substring(0, 120) + "...";
            }
            out.println("[audit] repair round " + round + " - " + line);
        }

        private void banner(String modelName, String maskedKey) {
            out.println("""
                    ============================================================
                     JAVA CONCEPT & CODE MENTOR
                     Concept -> Blueprint -> Compilable code, every time.
                    ============================================================""");
            out.println(" model: " + modelName + "   key: " + maskedKey);
            out.println(" Type a Java question, /help for commands, exit to quit.");
        }
    }
}