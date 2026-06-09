package com.mobilefinetuner.visualizer.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.FolderOpen
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.mobilefinetuner.visualizer.model.LocalModelAsset
import com.mobilefinetuner.visualizer.model.RunnerStatus
import com.mobilefinetuner.visualizer.model.RunnerUiState
import com.mobilefinetuner.visualizer.ui.theme.NeonAmber
import com.mobilefinetuner.visualizer.ui.theme.NeonBlue
import com.mobilefinetuner.visualizer.ui.theme.NeonGreen
import com.mobilefinetuner.visualizer.ui.theme.NeonRed
import com.mobilefinetuner.visualizer.viewmodel.ExperimentViewModel
import java.util.Locale

@Composable
fun RunnerScreen(
    runner: RunnerUiState,
    viewModel: ExperimentViewModel,
    modifier: Modifier = Modifier
) {
    val importLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        if (uri != null) viewModel.importModelDirectory(uri)
    }

    LazyColumn(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp)
    ) {
        item {
            RunnerHeader(runner)
        }
        item {
            ModelSection(
                runner = runner,
                onImport = { importLauncher.launch(null) },
                onRefresh = viewModel::refreshLocalModels,
                onSelect = viewModel::selectRunnerModel
            )
        }
        item {
            RunnerControls(
                runner = runner,
                onLoadWeights = viewModel::setRunnerLoadWeights,
                onSequenceLength = viewModel::setRunnerSequenceLength,
                onBatchSize = viewModel::setRunnerBatchSize,
                onGradientAccumulationSteps = viewModel::setRunnerGradientAccumulationSteps,
                onSteps = viewModel::setRunnerSteps,
                onTrainingText = viewModel::setRunnerTrainingText,
                onStart = viewModel::startRunnerTraining,
                onStop = viewModel::stopRunnerTraining
            )
        }
        if (runner.error != null) {
            item {
                ErrorBand(runner.error)
            }
        }
        item {
            ResultSection(runner)
        }
        item {
            LogSection(runner)
        }
    }
}

@Composable
private fun RunnerHeader(runner: RunnerUiState) {
    val color = when (runner.status) {
        RunnerStatus.RUNNING -> NeonBlue
        RunnerStatus.COMPLETED -> NeonGreen
        RunnerStatus.FAILED -> NeonRed
        RunnerStatus.IMPORTING, RunnerStatus.STOPPING -> NeonAmber
        RunnerStatus.IDLE, RunnerStatus.READY -> Color(0xFF8E8E93)
    }

    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.94f)),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Training Runner",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = runner.currentMessage,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                StatusBadge(label = runner.status.name.lowercase(Locale.US), color = color)
            }
            if (runner.status == RunnerStatus.RUNNING || runner.status == RunnerStatus.IMPORTING) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun ModelSection(
    runner: RunnerUiState,
    onImport: () -> Unit,
    onRefresh: () -> Unit,
    onSelect: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    val selected = runner.models.firstOrNull { it.id == runner.selectedModelId }

    Card(
        colors = CardDefaults.cardColors(containerColor = Color(0xFFFAFBFF)),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, Color(0xFFDDE2EC), RoundedCornerShape(18.dp))
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(onClick = onImport, modifier = Modifier.weight(1f)) {
                    Icon(Icons.Rounded.FolderOpen, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Import Model")
                }
                IconButton(onClick = onRefresh) {
                    Icon(Icons.Rounded.Refresh, contentDescription = "Refresh models")
                }
            }

            Box(modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(
                    onClick = { expanded = true },
                    enabled = runner.models.isNotEmpty(),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.Start) {
                        Text(
                            text = selected?.name ?: "No local model",
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(
                            text = selected?.let {
                                "${it.family}  ${formatBytes(it.sizeBytes)}  ${if (it.hasWeights) "weights" else "no weights"}"
                            } ?: "Import a HuggingFace model directory",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    Icon(Icons.Rounded.ExpandMore, contentDescription = null)
                }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    runner.models.forEach { model ->
                        DropdownMenuItem(
                            text = { ModelMenuRow(model) },
                            onClick = {
                                expanded = false
                                onSelect(model.id)
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ModelMenuRow(model: LocalModelAsset) {
    Column(modifier = Modifier.width(320.dp)) {
        Text(model.name, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Text(
            "${model.family}  ${formatBytes(model.sizeBytes)}  ${if (model.hasWeights) "weights" else "no weights"}",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun RunnerControls(
    runner: RunnerUiState,
    onLoadWeights: (Boolean) -> Unit,
    onSequenceLength: (Int) -> Unit,
    onBatchSize: (Int) -> Unit,
    onGradientAccumulationSteps: (Int) -> Unit,
    onSteps: (Int) -> Unit,
    onTrainingText: (String) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit
) {
    val running = runner.status == RunnerStatus.RUNNING || runner.status == RunnerStatus.IMPORTING
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.94f)),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, Color(0xFFE6E8EF), RoundedCornerShape(18.dp))
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text("Load SafeTensors", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "Use full pretrained weights",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = runner.loadWeights,
                    onCheckedChange = onLoadWeights,
                    enabled = !running
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                OutlinedTextField(
                    value = runner.sequenceLength.toString(),
                    onValueChange = { text -> text.toIntOrNull()?.let(onSequenceLength) },
                    enabled = !running,
                    label = { Text("Seq length") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.weight(1f),
                    singleLine = true
                )
                OutlinedTextField(
                    value = runner.batchSize.toString(),
                    onValueChange = { text -> text.toIntOrNull()?.let(onBatchSize) },
                    enabled = !running,
                    label = { Text("Batch size") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.weight(1f),
                    singleLine = true
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                OutlinedTextField(
                    value = runner.steps.toString(),
                    onValueChange = { text -> text.toIntOrNull()?.let(onSteps) },
                    enabled = !running,
                    label = { Text("Steps") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.weight(1f),
                    singleLine = true
                )
                OutlinedTextField(
                    value = runner.gradientAccumulationSteps.toString(),
                    onValueChange = { text -> text.toIntOrNull()?.let(onGradientAccumulationSteps) },
                    enabled = !running,
                    label = { Text("Grad accum") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.weight(1f),
                    singleLine = true
                )
            }

            OutlinedTextField(
                value = runner.trainingText,
                onValueChange = onTrainingText,
                enabled = !running,
                label = { Text("Training text") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 3,
                maxLines = 5
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Button(
                    onClick = onStart,
                    enabled = !running,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Rounded.PlayArrow, contentDescription = null)
                    Spacer(Modifier.width(6.dp))
                    Text("Start")
                }
                OutlinedButton(
                    onClick = onStop,
                    enabled = runner.status == RunnerStatus.RUNNING,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Rounded.Stop, contentDescription = null)
                    Spacer(Modifier.width(6.dp))
                    Text("Stop")
                }
            }
        }
    }
}

@Composable
private fun ResultSection(runner: RunnerUiState) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Color(0xFFFAFBFF)),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Step Results", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            if (runner.results.isEmpty()) {
                Text(
                    "No completed step",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else {
                runner.results.takeLast(8).forEach { result ->
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Step ${result.step}", fontWeight = FontWeight.Medium)
                            Text("${result.elapsedMs} ms", color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                "loss=${String.format(Locale.US, "%.5f", result.loss)}",
                                fontFamily = FontFamily.Monospace
                            )
                            Text(
                                if (result.optimizerStep) "Adam" else "Accum ${result.accumulationStep}/${result.gradientAccumulationSteps}",
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun LogSection(runner: RunnerUiState) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Color(0xFF101418)),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Runner Log", style = MaterialTheme.typography.titleMedium, color = Color.White)
            HorizontalDivider(color = Color.White.copy(alpha = 0.12f))
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                val lines = if (runner.logs.isEmpty()) listOf("idle") else runner.logs.takeLast(80)
                lines.forEach { line ->
                    Text(
                        text = line,
                        color = Color(0xFFE7EEF8),
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
            Spacer(Modifier.height(2.dp))
        }
    }
}

@Composable
private fun ErrorBand(message: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFFFFECEB), RoundedCornerShape(12.dp))
            .padding(12.dp)
    ) {
        Text(message, color = NeonRed, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun StatusBadge(label: String, color: Color) {
    Text(
        text = label,
        style = MaterialTheme.typography.labelLarge,
        color = color,
        modifier = Modifier
            .background(color.copy(alpha = 0.12f), RoundedCornerShape(8.dp))
            .padding(horizontal = 9.dp, vertical = 5.dp)
    )
}

private fun formatBytes(bytes: Long): String {
    if (bytes <= 0L) return "0 MB"
    val mb = bytes / (1024.0 * 1024.0)
    return if (mb < 1024.0) {
        String.format(Locale.US, "%.1f MB", mb)
    } else {
        String.format(Locale.US, "%.2f GB", mb / 1024.0)
    }
}
