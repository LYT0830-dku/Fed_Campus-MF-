package com.mobilefinetuner.visualizer.viewmodel

import android.app.Application
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilefinetuner.sdk.MobileFineTuner
import com.mobilefinetuner.visualizer.data.TrainingLogParser
import com.mobilefinetuner.visualizer.model.DashboardTab
import com.mobilefinetuner.visualizer.model.DashboardUiState
import com.mobilefinetuner.visualizer.model.EventType
import com.mobilefinetuner.visualizer.model.LocalModelAsset
import com.mobilefinetuner.visualizer.model.RssPoint
import com.mobilefinetuner.visualizer.model.RunEvent
import com.mobilefinetuner.visualizer.model.RunHandle
import com.mobilefinetuner.visualizer.model.RunSnapshot
import com.mobilefinetuner.visualizer.model.RunStatus
import com.mobilefinetuner.visualizer.model.RunSummary
import com.mobilefinetuner.visualizer.model.RunnerStatus
import com.mobilefinetuner.visualizer.model.RunnerStepResult
import com.mobilefinetuner.visualizer.model.StepMetric
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

class ExperimentViewModel(application: Application) : AndroidViewModel(application) {

    private val resolver = application.contentResolver
    private val prefs = application.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState.asStateFlow()

    private var pollingJob: Job? = null
    private var runnerJob: Job? = null
    @Volatile private var stopRunnerRequested: Boolean = false
    private var demoCursor: Int = DEMO_START_CURSOR

    init {
        refreshLocalModels()
        val rawUri = prefs.getString(KEY_LAST_ROOT_URI, null)
        if (!rawUri.isNullOrBlank()) {
            val uri = Uri.parse(rawUri)
            _uiState.value = _uiState.value.copy(selectedRootUri = uri)
            scanRuns(uri)
        }
    }

    fun refreshLocalModels() {
        viewModelScope.launch(Dispatchers.IO) {
            val models = scanLocalModelAssets()
            val current = _uiState.value.runner.selectedModelId
            val selected = when {
                models.isEmpty() -> null
                current != null && models.any { it.id == current } -> current
                else -> models.first().id
            }
            _uiState.value = _uiState.value.copy(
                runner = _uiState.value.runner.copy(
                    models = models,
                    selectedModelId = selected,
                    status = if (models.isEmpty()) RunnerStatus.IDLE else _uiState.value.runner.status,
                    currentMessage = if (models.isEmpty()) "Import a model to start" else "Ready",
                    error = null
                )
            )
        }
    }

    fun importModelDirectory(uri: Uri) {
        try {
            resolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (_: SecurityException) {
            // Some providers cannot persist permissions; the immediate import can still proceed.
        }

        viewModelScope.launch(Dispatchers.IO) {
            val root = DocumentFile.fromTreeUri(getApplication(), uri)
            if (root == null || !root.exists() || !root.isDirectory) {
                updateRunnerError("Cannot open selected model directory.")
                return@launch
            }

            val modelName = sanitizeModelName(root.name ?: "model")
            val modelRoot = File(getApplication<Application>().filesDir, LOCAL_MODELS_DIR)
            val dest = File(modelRoot, modelName)

            _uiState.value = _uiState.value.copy(
                runner = _uiState.value.runner.copy(
                    status = RunnerStatus.IMPORTING,
                    importingModelName = modelName,
                    currentMessage = "Importing $modelName",
                    error = null
                )
            )

            try {
                deleteFileTree(dest)
                if (!dest.mkdirs()) {
                    throw IllegalStateException("Cannot create model directory: ${dest.absolutePath}")
                }
                copyDocumentTree(root, dest)
                val config = File(dest, "config.json")
                if (!config.isFile) {
                    throw IllegalStateException("Imported directory does not contain config.json.")
                }

                val models = scanLocalModelAssets()
                _uiState.value = _uiState.value.copy(
                    runner = _uiState.value.runner.copy(
                        status = RunnerStatus.READY,
                        models = models,
                        selectedModelId = models.firstOrNull { it.path == dest.absolutePath }?.id
                            ?: models.firstOrNull()?.id,
                        importingModelName = null,
                        currentMessage = "Imported $modelName",
                        logs = (_uiState.value.runner.logs + "Imported $modelName").takeLast(MAX_RUNNER_LOGS),
                        error = null
                    )
                )
            } catch (t: Throwable) {
                deleteFileTree(dest)
                updateRunnerError("Import failed: ${t.message ?: t::class.java.simpleName}")
            }
        }
    }

    fun selectRunnerModel(modelId: String) {
        _uiState.value = _uiState.value.copy(
            runner = _uiState.value.runner.copy(selectedModelId = modelId, error = null)
        )
    }

    fun setRunnerLoadWeights(loadWeights: Boolean) {
        _uiState.value = _uiState.value.copy(
            runner = _uiState.value.runner.copy(loadWeights = loadWeights)
        )
    }

    fun setRunnerSequenceLength(sequenceLength: Int) {
        _uiState.value = _uiState.value.copy(
            runner = _uiState.value.runner.copy(sequenceLength = sequenceLength.coerceIn(2, 512))
        )
    }

    fun setRunnerSteps(steps: Int) {
        _uiState.value = _uiState.value.copy(
            runner = _uiState.value.runner.copy(steps = steps.coerceIn(1, 100))
        )
    }

    fun setRunnerTrainingText(text: String) {
        _uiState.value = _uiState.value.copy(
            runner = _uiState.value.runner.copy(trainingText = text)
        )
    }

    fun startRunnerTraining() {
        if (runnerJob?.isActive == true) return
        val state = _uiState.value.runner
        val model = state.models.firstOrNull { it.id == state.selectedModelId }
        if (model == null) {
            updateRunnerError("Select or import a model first.")
            return
        }
        if (state.loadWeights && !model.hasWeights) {
            updateRunnerError("Selected model has no SafeTensors weights. Disable weight loading or import weights.")
            return
        }
        if (state.trainingText.isBlank()) {
            updateRunnerError("Training text must not be empty.")
            return
        }

        stopRunnerRequested = false
        runnerJob = viewModelScope.launch(Dispatchers.Default) {
            _uiState.value = _uiState.value.copy(
                activeTab = DashboardTab.RUNNER,
                runner = _uiState.value.runner.copy(
                    status = RunnerStatus.RUNNING,
                    currentMessage = "Opening ${model.name}",
                    results = emptyList(),
                    logs = listOf("Opening ${model.name}"),
                    error = null
                )
            )

            try {
                MobileFineTuner.open(model.path, state.loadWeights).use { mf ->
                    appendRunnerLog("Model opened: ${model.family}, loadWeights=${state.loadWeights}")
                    mf.initLora(MobileFineTuner.LoraConfig.attentionQkvo())
                    appendRunnerLog("LoRA initialized: q_proj/k_proj/v_proj/o_proj")
                    mf.createTrainer(MobileFineTuner.TrainerConfig(2e-4f, 0.0f, 1.0f, -100))
                    appendRunnerLog("Trainer created")

                    for (step in 1..state.steps) {
                        if (stopRunnerRequested) break
                        updateRunnerMessage("Running step $step/${state.steps}")
                        val t0 = SystemClock.elapsedRealtime()
                        val result = mf.trainTextBatch(
                            arrayOf(state.trainingText),
                            state.sequenceLength,
                            true
                        )
                        val elapsed = SystemClock.elapsedRealtime() - t0
                        val record = RunnerStepResult(
                            step = step,
                            loss = result.loss,
                            trainableTensorCount = result.trainableTensorCount,
                            validLabelCount = result.validLabelCount,
                            elapsedMs = elapsed
                        )
                        _uiState.value = _uiState.value.copy(
                            runner = _uiState.value.runner.copy(
                                results = _uiState.value.runner.results + record
                            )
                        )
                        appendRunnerLog(
                            "step=$step loss=${String.format(Locale.US, "%.6f", result.loss)} " +
                                "validLabels=${result.validLabelCount} elapsedMs=$elapsed"
                        )
                    }
                }

                val stopped = stopRunnerRequested
                _uiState.value = _uiState.value.copy(
                    runner = _uiState.value.runner.copy(
                        status = if (stopped) RunnerStatus.IDLE else RunnerStatus.COMPLETED,
                        currentMessage = if (stopped) "Stopped" else "Completed",
                        error = null
                    )
                )
            } catch (t: Throwable) {
                _uiState.value = _uiState.value.copy(
                    runner = _uiState.value.runner.copy(
                        status = RunnerStatus.FAILED,
                        currentMessage = "Failed",
                        error = t.message ?: t::class.java.simpleName
                    )
                )
                appendRunnerLog("Failed: ${t.message ?: t::class.java.simpleName}")
            }
        }
    }

    fun stopRunnerTraining() {
        stopRunnerRequested = true
        _uiState.value = _uiState.value.copy(
            runner = _uiState.value.runner.copy(
                status = RunnerStatus.STOPPING,
                currentMessage = "Stopping after current native call"
            )
        )
    }

    fun selectTab(tab: DashboardTab) {
        _uiState.value = _uiState.value.copy(activeTab = tab)
    }

    fun onDirectoryPicked(uri: Uri) {
        try {
            resolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (_: SecurityException) {
            // Some providers cannot persist permissions; still try to read immediately.
        }

        prefs.edit().putString(KEY_LAST_ROOT_URI, uri.toString()).apply()
        _uiState.value = _uiState.value.copy(
            selectedRootUri = uri,
            error = null,
            isLoading = true,
            runHandles = emptyList(),
            selectedRunId = null,
            compareRunId = null,
            snapshot = null,
            compareSnapshot = null
        )
        scanRuns(uri)
    }

    fun refreshNow() {
        val root = _uiState.value.selectedRootUri ?: return
        scanRuns(root)
    }

    fun enableDemoMode() {
        val demoRuns = listOf(
            RunHandle(
                id = DEMO_PRIMARY_ID,
                name = "demo/primary_gpt2",
                dirUri = Uri.EMPTY,
                trainLogUri = Uri.EMPTY,
                rssCsvUri = null,
                metricsNdjsonUri = null,
                lastModifiedMs = System.currentTimeMillis()
            ),
            RunHandle(
                id = DEMO_COMPARE_ID,
                name = "demo/compare_gemma",
                dirUri = Uri.EMPTY,
                trainLogUri = Uri.EMPTY,
                rssCsvUri = null,
                metricsNdjsonUri = null,
                lastModifiedMs = System.currentTimeMillis()
            )
        )
        demoCursor = DEMO_START_CURSOR
        _uiState.value = _uiState.value.copy(
            activeTab = DashboardTab.OVERVIEW,
            runHandles = demoRuns,
            selectedRunId = DEMO_PRIMARY_ID,
            compareRunId = DEMO_COMPARE_ID,
            snapshot = null,
            compareSnapshot = null,
            isLoading = false,
            error = null
        )
        startPolling()
    }

    fun selectRun(runId: String) {
        _uiState.value = _uiState.value.copy(selectedRunId = runId)
        startPolling()
    }

    fun selectComparisonRun(runId: String?) {
        _uiState.value = _uiState.value.copy(compareRunId = runId)
        startPolling()
    }

    private fun scanRuns(rootUri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val root = DocumentFile.fromTreeUri(getApplication(), rootUri)
            if (root == null || !root.exists() || !root.isDirectory) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Cannot open selected directory."
                )
                return@launch
            }

            val runs = discoverRuns(root)
            val selectedId = _uiState.value.selectedRunId
            val activeId = when {
                runs.isEmpty() -> null
                selectedId != null && runs.any { it.id == selectedId } -> selectedId
                else -> runs.first().id
            }
            val existingCompareId = _uiState.value.compareRunId
            val compareId = when {
                runs.isEmpty() -> null
                existingCompareId != null && runs.any { it.id == existingCompareId } &&
                    existingCompareId != activeId -> existingCompareId
                else -> runs.firstOrNull { it.id != activeId }?.id
            }

            _uiState.value = _uiState.value.copy(
                isLoading = false,
                runHandles = runs,
                selectedRunId = activeId,
                compareRunId = compareId,
                compareSnapshot = null,
                error = if (runs.isEmpty()) "No train.log found under this folder." else null
            )

            startPolling()
        }
    }

    private fun startPolling() {
        pollingJob?.cancel()
        val runId = _uiState.value.selectedRunId ?: return

        pollingJob = viewModelScope.launch(Dispatchers.IO) {
            if (runId.startsWith(DEMO_PREFIX)) {
                while (true) {
                    val primary = buildDemoSnapshot(DEMO_PRIMARY_ID, "demo/primary_gpt2", isCompare = false, cursor = demoCursor)
                    val compare = buildDemoSnapshot(DEMO_COMPARE_ID, "demo/compare_gemma", isCompare = true, cursor = demoCursor)
                    _uiState.value = _uiState.value.copy(
                        snapshot = primary,
                        compareSnapshot = compare
                    )
                    demoCursor = if (demoCursor >= DEMO_TOTAL_STEPS) DEMO_START_CURSOR else demoCursor + 1
                    delay(DEMO_TICK_MS)
                }
            }

            while (true) {
                val state = _uiState.value
                val run = state.runHandles.firstOrNull { it.id == runId } ?: break
                val snapshot = parseRunSnapshot(run)

                val compareSnapshot = state.compareRunId
                    ?.takeIf { it != runId }
                    ?.let { compareId ->
                        state.runHandles.firstOrNull { it.id == compareId }
                    }
                    ?.let { parseRunSnapshot(it) }

                _uiState.value = _uiState.value.copy(
                    snapshot = snapshot,
                    compareSnapshot = compareSnapshot
                )

                val delayMs = if (snapshot.status == RunStatus.RUNNING) POLL_FAST_MS else POLL_SLOW_MS
                delay(delayMs)
            }
        }
    }

    private fun discoverRuns(root: DocumentFile): List<RunHandle> {
        val runs = mutableListOf<RunHandle>()

        fun walk(dir: DocumentFile, relPath: String, depth: Int) {
            if (depth > MAX_SCAN_DEPTH || !dir.isDirectory) return

            val children = runCatching { dir.listFiles().toList() }.getOrDefault(emptyList())
            if (children.isEmpty()) return

            val train = children.firstOrNull { it.isFile && it.name == "train.log" }
            if (train != null) {
                val rss = children.firstOrNull { it.isFile && it.name == "rss.csv" }
                val ndjson = children.firstOrNull {
                    it.isFile && (it.name == "metrics.ndjson" || it.name == "metrics.jsonl")
                }
                val runName = if (relPath.isBlank()) {
                    dir.name ?: "run"
                } else {
                    relPath
                }
                val lastModified = listOfNotNull(
                    train.lastModified(),
                    rss?.lastModified(),
                    ndjson?.lastModified()
                ).maxOrNull() ?: 0L

                runs += RunHandle(
                    id = dir.uri.toString(),
                    name = runName,
                    dirUri = dir.uri,
                    trainLogUri = train.uri,
                    rssCsvUri = rss?.uri,
                    metricsNdjsonUri = ndjson?.uri,
                    lastModifiedMs = lastModified
                )
            }

            children
                .filter { it.isDirectory }
                .forEach { child ->
                    val childName = child.name ?: "folder"
                    val nextPath = if (relPath.isBlank()) childName else "$relPath/$childName"
                    walk(child, nextPath, depth + 1)
                }
        }

        walk(root, "", 0)
        return runs.sortedByDescending { it.lastModifiedMs }
    }

    private suspend fun parseRunSnapshot(run: RunHandle): RunSnapshot = withContext(Dispatchers.IO) {
        val logLines = readLastLines(run.trainLogUri, MAX_LOG_LINES)
        val parsedLog = TrainingLogParser.parseTrainLog(logLines)

        val rssPoints = run.rssCsvUri
            ?.let { readLastLines(it, MAX_RSS_LINES) }
            ?.let { TrainingLogParser.parseRssCsv(it) }
            ?: emptyList()

        val ndjsonMetrics = run.metricsNdjsonUri
            ?.let { readLastLines(it, MAX_NDJSON_LINES) }
            ?.let { TrainingLogParser.parseMetricsNdjson(it) }
            ?: emptyList()

        val mergedMetrics = mergeMetrics(parsedLog.metrics, ndjsonMetrics)

        RunSnapshot(
            runId = run.id,
            runName = run.name,
            status = if (ndjsonMetrics.isNotEmpty() && parsedLog.status == RunStatus.IDLE) {
                RunStatus.RUNNING
            } else {
                parsedLog.status
            },
            metrics = mergedMetrics,
            rssPoints = rssPoints,
            events = parsedLog.events.takeLast(MAX_EVENT_LINES),
            logTail = logLines.takeLast(MAX_LOG_TAIL_LINES),
            summary = parsedLog.summary,
            updatedAtMs = System.currentTimeMillis()
        )
    }

    private fun mergeMetrics(logMetrics: List<StepMetric>, ndjsonMetrics: List<StepMetric>): List<StepMetric> {
        if (ndjsonMetrics.isEmpty()) return logMetrics
        if (logMetrics.isEmpty()) return ndjsonMetrics

        val map = linkedMapOf<Int, StepMetric>()
        for (m in logMetrics) {
            map[m.step] = m
        }
        for (m in ndjsonMetrics) {
            val cur = map[m.step]
            map[m.step] = if (cur == null) {
                m
            } else {
                StepMetric(
                    step = m.step,
                    totalSteps = m.totalSteps ?: cur.totalSteps,
                    loss = m.loss ?: cur.loss,
                    ppl = m.ppl ?: cur.ppl,
                    lr = m.lr ?: cur.lr,
                    tokens = m.tokens ?: cur.tokens,
                    source = "merge"
                )
            }
        }
        return map.values.sortedBy { it.step }
    }

    private fun buildDemoSnapshot(
        runId: String,
        runName: String,
        isCompare: Boolean,
        cursor: Int
    ): RunSnapshot {
        val maxStep = cursor.coerceIn(DEMO_START_CURSOR, DEMO_TOTAL_STEPS)
        val metrics = (1..maxStep).map { step ->
            val baseLoss = if (isCompare) 5.2 else 5.0
            val speed = if (isCompare) 0.021 else 0.025
            val oscillation = if (isCompare) 0.24 else 0.18
            val loss = (baseLoss - speed * step + oscillation * kotlin.math.sin(step / 7.2)).coerceAtLeast(1.2)
            val ppl = kotlin.math.exp(loss / 2.15).coerceAtMost(2200.0)
            val lr = (2e-4 * (1.0 - (step / DEMO_TOTAL_STEPS.toDouble()) * 0.82)).coerceAtLeast(2.0e-5)
            val tokens = 1024 + (if (isCompare) 128 else 0)
            StepMetric(
                step = step,
                totalSteps = DEMO_TOTAL_STEPS,
                loss = loss,
                ppl = ppl,
                lr = lr,
                tokens = tokens,
                source = "demo"
            )
        }

        val rss = (0 until maxStep).map { i ->
            val trend = if (isCompare) 1940.0 else 1820.0
            val ramp = i * if (isCompare) 0.9 else 0.7
            val wobble = kotlin.math.sin(i / 4.0) * 21.0
            RssPoint(
                index = i,
                rssMb = trend + ramp + wobble,
                timeLabel = null
            )
        }

        val evalPpl = formatDemoValue("%.2f", metrics.lastOrNull()?.ppl)
        val events = listOf(
            RunEvent(step = maxStep, type = EventType.INFO, message = "[Train] step=$maxStep running", raw = "demo"),
            RunEvent(step = maxStep, type = EventType.RSS, message = "[RSS] ${(rss.lastOrNull()?.rssMb ?: 0.0).toInt()} MB", raw = "demo"),
            RunEvent(step = maxStep, type = EventType.EVAL, message = "[Eval] ppl=$evalPpl", raw = "demo")
        )

        val status = if (maxStep >= DEMO_TOTAL_STEPS) RunStatus.COMPLETED else RunStatus.RUNNING
        val summary = RunSummary(
            maxTrainRssMb = rss.maxOfOrNull { it.rssMb },
            finalEmaLoss = metrics.lastOrNull()?.loss,
            totalSteps = DEMO_TOTAL_STEPS,
            totalTokens = metrics.sumOf { it.tokens?.toLong() ?: 0L }
        )

        return RunSnapshot(
            runId = runId,
            runName = runName,
            status = status,
            metrics = metrics,
            rssPoints = rss,
            events = events,
            logTail = buildList {
                val tail = metrics.takeLast(90)
                tail.forEachIndexed { idx, m ->
                    add("[Train] step=${m.step}/${m.totalSteps} | loss=${formatDemoValue("%.3f", m.loss)} | ppl=${formatDemoValue("%.2f", m.ppl)} | lr=${formatDemoValue("%.6f", m.lr)}")
                    // Interleave RSS every 10 steps
                    if (idx % 10 == 9) {
                        val rssMb = (if (isCompare) 1940.0 + idx * 0.9 else 1820.0 + idx * 0.7)
                            .coerceAtMost(2200.0)
                        add("[RSS] step=${m.step} rss=${rssMb.toInt()} MB")
                    }
                    // Interleave Eval every 20 steps
                    if (idx % 20 == 19) {
                        val logEvalPpl = formatDemoValue("%.2f", m.ppl)
                        add("[Eval] step=${m.step} ppl=$logEvalPpl")
                    }
                    // Interleave Checkpoint every 30 steps
                    if (idx % 30 == 29) {
                        add("[Checkpoint] step=${m.step} saved checkpoint to /data/ckpt/step_${m.step}.bin")
                    }
                }
            },
            summary = summary,
            updatedAtMs = System.currentTimeMillis()
        )
    }

    private fun readLastLines(uri: Uri, maxLines: Int): List<String> {
        val deque = ArrayDeque<String>(maxLines)
        resolver.openInputStream(uri)?.bufferedReader()?.useLines { sequence ->
            sequence.forEach { line ->
                if (deque.size == maxLines) {
                    deque.removeFirst()
                }
                deque.addLast(line)
            }
        }
        return deque.toList()
    }

    private fun formatDemoValue(pattern: String, value: Double?): String {
        return value?.let { String.format(Locale.US, pattern, it) } ?: "N/A"
    }

    private fun scanLocalModelAssets(): List<LocalModelAsset> {
        val root = File(getApplication<Application>().filesDir, LOCAL_MODELS_DIR)
        if (!root.exists()) return emptyList()
        return root.listFiles()
            ?.filter { it.isDirectory && File(it, "config.json").isFile }
            ?.map { dir ->
                LocalModelAsset(
                    id = dir.absolutePath,
                    name = dir.name,
                    path = dir.absolutePath,
                    family = inferFamilyFromConfig(File(dir, "config.json")),
                    hasWeights = dir.walkTopDown().any { file ->
                        file.isFile && (
                            file.name.endsWith(".safetensors") ||
                                file.name == "model.safetensors.index.json"
                            )
                    },
                    sizeBytes = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() },
                    updatedAtMs = dir.walkTopDown().filter { it.isFile }.map { it.lastModified() }.maxOrNull()
                        ?: dir.lastModified()
                )
            }
            ?.sortedBy { it.name.lowercase(Locale.US) }
            ?: emptyList()
    }

    private fun inferFamilyFromConfig(configFile: File): String {
        return runCatching {
            val json = JSONObject(configFile.readText())
            when (val type = json.optString("model_type", "").lowercase(Locale.US)) {
                "gpt2" -> "GPT-2"
                "gemma3", "gemma" -> "Gemma"
                "qwen2", "qwen3" -> "Qwen"
                "llama" -> "Llama"
                else -> if (type.isBlank()) "Unknown" else type.uppercase(Locale.US)
            }
        }.getOrDefault("Unknown")
    }

    private fun copyDocumentTree(source: DocumentFile, dest: File) {
        val children = source.listFiles()
        for (child in children) {
            val name = child.name ?: continue
            if (name == ".cache" || name.endsWith(".lock")) continue
            val target = File(dest, name)
            if (child.isDirectory) {
                if (!target.exists() && !target.mkdirs()) {
                    throw IllegalStateException("Cannot create directory: ${target.absolutePath}")
                }
                copyDocumentTree(child, target)
            } else if (child.isFile) {
                resolver.openInputStream(child.uri)?.use { input ->
                    FileOutputStream(target).use { output ->
                        input.copyTo(output)
                    }
                } ?: throw IllegalStateException("Cannot read file: $name")
            }
        }
    }

    private fun sanitizeModelName(raw: String): String {
        return raw.trim()
            .replace(Regex("[^A-Za-z0-9._-]+"), "_")
            .trim('_')
            .ifBlank { "model" }
    }

    private fun deleteFileTree(file: File) {
        if (!file.exists()) return
        if (file.isDirectory) {
            file.listFiles()?.forEach { deleteFileTree(it) }
        }
        file.delete()
    }

    private fun appendRunnerLog(message: String) {
        _uiState.value = _uiState.value.copy(
            runner = _uiState.value.runner.copy(
                logs = (_uiState.value.runner.logs + message).takeLast(MAX_RUNNER_LOGS),
                currentMessage = message
            )
        )
    }

    private fun updateRunnerMessage(message: String) {
        _uiState.value = _uiState.value.copy(
            runner = _uiState.value.runner.copy(currentMessage = message)
        )
    }

    private fun updateRunnerError(message: String) {
        _uiState.value = _uiState.value.copy(
            runner = _uiState.value.runner.copy(
                status = RunnerStatus.FAILED,
                currentMessage = "Failed",
                error = message,
                importingModelName = null,
                logs = (_uiState.value.runner.logs + message).takeLast(MAX_RUNNER_LOGS)
            )
        )
    }

    companion object {
        private const val PREFS_NAME = "mft_visualizer"
        private const val KEY_LAST_ROOT_URI = "last_root_uri"
        private const val LOCAL_MODELS_DIR = "models"

        private const val MAX_SCAN_DEPTH = 10
        private const val MAX_LOG_LINES = 20000
        private const val MAX_LOG_TAIL_LINES = 1000
        private const val MAX_RSS_LINES = 12000
        private const val MAX_NDJSON_LINES = 20000
        private const val MAX_EVENT_LINES = 500
        private const val MAX_RUNNER_LOGS = 500

        private const val POLL_FAST_MS = 1500L
        private const val POLL_SLOW_MS = 3500L

        private const val DEMO_PREFIX = "demo:"
        private const val DEMO_PRIMARY_ID = "${DEMO_PREFIX}primary"
        private const val DEMO_COMPARE_ID = "${DEMO_PREFIX}compare"
        private const val DEMO_TOTAL_STEPS = 220
        private const val DEMO_START_CURSOR = 24
        private const val DEMO_TICK_MS = 1200L
    }
}
