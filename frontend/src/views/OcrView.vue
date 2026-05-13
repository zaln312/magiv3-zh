<template>
  <div class="ocr-page">
    <div class="page-header">
      <h2>OCR 识别结果</h2>
      <div class="header-actions">
        <el-select v-model="currentImgIdx" placeholder="选择图片" style="width: 200px">
          <el-option
            v-for="(_, idx) in ocrResults"
            :key="idx"
            :label="`图片 ${idx + 1}`"
            :value="idx"
          />
        </el-select>
        <el-button type="primary" @click="runPredict" :loading="predictLoading">
          执行 Predict
        </el-button>
      </div>
    </div>

    <div v-if="ocrResults.length === 0" class="empty-state">
      <el-empty description="暂无 OCR 结果，请先执行 OCR 识别" />
    </div>

    <div v-else class="ocr-workspace">
      <div class="canvas-container" ref="canvasContainer">
        <v-stage ref="stageRef" :config="stageConfig" @mousedown="handleStageMouseDown">
          <v-layer>
            <v-image :config="imageConfig" />
            <v-rect
              v-for="(box, idx) in currentBoxes"
              :key="'box-' + idx"
              :config="getBoxConfig(idx, box)"
            />
            <v-line
              v-for="(box, idx) in currentBoxes"
              :key="'line-' + idx"
              :config="getLineConfig(idx, box)"
            />
            <v-text
              v-for="(box, idx) in currentBoxes"
              :key="'text-' + idx"
              :config="getTextConfig(idx)"
            />
          </v-layer>
        </v-stage>
      </div>

      <div class="text-panel">
        <h3>文本列表</h3>
        <div
          v-for="(text, idx) in currentTexts"
          :key="idx"
          class="text-item"
          :class="{ active: selectedBoxIdx === idx }"
          @click="selectBox(idx)"
        >
          <span class="text-index">{{ idx }}</span>
          <el-input
            v-model="currentTexts[idx]"
            size="small"
            @change="updateText(idx, currentTexts[idx])"
          />
          <el-button
            type="danger"
            size="small"
            circle
            :icon="Delete"
            @click.stop="deleteBox(idx)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Delete } from '@element-plus/icons-vue'
import { ocrApi, predictApi } from '../api/endpoints'

const router = useRouter()

interface OcrResult {
  img_path: string
  boxes: number[][]
  texts: string[]
}

const ocrResults = ref<OcrResult[]>([])
const currentImgIdx = ref(0)
const selectedBoxIdx = ref<number | null>(null)
const predictLoading = ref(false)

const stageRef = ref<any>(null)
const canvasContainer = ref<HTMLElement | null>(null)

const imageObj = ref<HTMLImageElement | null>(null)
const canvasWidth = ref(900)
const canvasHeight = ref(600)
const imageNaturalWidth = ref(1)
const imageNaturalHeight = ref(1)
const scaleX = ref(1)
const scaleY = ref(1)

const currentBoxes = computed(() => {
  if (currentImgIdx.value >= ocrResults.value.length) return []
  return ocrResults.value[currentImgIdx.value].boxes || []
})

const currentTexts = computed({
  get: () => {
    if (currentImgIdx.value >= ocrResults.value.length) return []
    return ocrResults.value[currentImgIdx.value].texts || []
  },
  set: (val) => {
    if (currentImgIdx.value < ocrResults.value.length) {
      ocrResults.value[currentImgIdx.value].texts = val
    }
  },
})

const stageConfig = computed(() => ({
  width: canvasWidth.value,
  height: canvasHeight.value,
}))

const imageConfig = computed(() => ({
  image: imageObj.value,
  width: canvasWidth.value,
  height: canvasHeight.value,
}))

function toCanvasX(x: number) {
  return x * scaleX.value
}

function toCanvasY(y: number) {
  return y * scaleY.value
}

function getBoxConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0])
  const y = toCanvasY(box[1])
  const w = toCanvasX(box[2]) - x
  const h = toCanvasY(box[3]) - y
  return {
    x, y, width: w, height: h,
    stroke: selectedBoxIdx.value === idx ? '#409eff' : '#00ff00',
    strokeWidth: selectedBoxIdx.value === idx ? 2 : 1,
    fill: 'rgba(0, 255, 0, 0.05)',
    draggable: false,
    name: 'box-' + idx,
  }
}

function getLineConfig(idx: number, box: number[]) {
  const cx = toCanvasX((box[0] + box[2]) / 2)
  const cy = toCanvasY((box[1] + box[3]) / 2)
  const textX = canvasWidth.value + 10
  const textY = 20 + idx * 40
  return {
    points: [cx, cy, textX, textY],
    stroke: '#999',
    strokeWidth: 1,
    dash: [4, 4],
    name: 'line-' + idx,
  }
}

function getTextConfig(idx: number) {
  const text = currentTexts.value[idx] || ''
  const displayText = text.length > 15 ? text.slice(0, 15) + '...' : text
  return {
    x: canvasWidth.value + 10,
    y: 20 + idx * 40,
    text: `[${idx}] ${displayText}`,
    fontSize: 13,
    fill: selectedBoxIdx.value === idx ? '#409eff' : '#333',
    name: 'text-' + idx,
  }
}

function selectBox(idx: number) {
  selectedBoxIdx.value = idx
}

function handleStageMouseDown(e: any) {
  const name = e.target?.name?.()
  if (name && name.startsWith('box-')) {
    const idx = parseInt(name.split('-')[1])
    selectBox(idx)
  } else if (name && name.startsWith('text-')) {
    const idx = parseInt(name.split('-')[1])
    selectBox(idx)
  }
}

async function updateText(idx: number, text: string) {
  try {
    await ocrApi.updateText(currentImgIdx.value, idx, text)
  } catch {
    ElMessage.error('更新文本失败')
  }
}

async function deleteBox(idx: number) {
  try {
    await ocrApi.deleteBox(currentImgIdx.value, idx)
    ocrResults.value[currentImgIdx.value].boxes.splice(idx, 1)
    ocrResults.value[currentImgIdx.value].texts.splice(idx, 1)
    if (selectedBoxIdx.value === idx) selectedBoxIdx.value = null
    ElMessage.success('已删除')
  } catch {
    ElMessage.error('删除失败')
  }
}

async function loadImage() {
  if (ocrResults.value.length === 0) return
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.src = `/api/images/serve/${currentImgIdx.value}?t=${Date.now()}`
  img.onload = () => {
    imageNaturalWidth.value = img.naturalWidth
    imageNaturalHeight.value = img.naturalHeight
    imageObj.value = img

    const containerWidth = canvasContainer.value?.clientWidth || 900
    const maxHeight = 700
    let w = containerWidth - 200
    let h = (img.naturalHeight / img.naturalWidth) * w
    if (h > maxHeight) {
      h = maxHeight
      w = (img.naturalWidth / img.naturalHeight) * h
    }
    canvasWidth.value = Math.floor(w)
    canvasHeight.value = Math.floor(h)
    scaleX.value = w / img.naturalWidth
    scaleY.value = h / img.naturalHeight
  }
}

async function loadOcrResults() {
  try {
    const res = await ocrApi.results()
    ocrResults.value = res.data.ocr_results || []
    if (ocrResults.value.length > 0) {
      await nextTick()
      await loadImage()
    }
  } catch {
    ocrResults.value = []
  }
}

async function runPredict() {
  predictLoading.value = true
  try {
    await predictApi.run()
    ElMessage.success('Predict 完成')
    router.push('/predict')
  } catch (e: any) {
    ElMessage.error('Predict 失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    predictLoading.value = false
  }
}

watch(currentImgIdx, async () => {
  selectedBoxIdx.value = null
  await nextTick()
  await loadImage()
})

onMounted(loadOcrResults)
</script>

<style scoped>
.ocr-page {
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.page-header h2 {
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

.empty-state {
  margin-top: 60px;
}

.ocr-workspace {
  display: flex;
  gap: 16px;
}

.canvas-container {
  flex: 1;
  overflow: auto;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background: #f5f5f5;
}

.text-panel {
  width: 280px;
  flex-shrink: 0;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 12px;
  max-height: 700px;
  overflow-y: auto;
}

.text-panel h3 {
  margin: 0 0 12px 0;
  font-size: 15px;
}

.text-item {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  padding: 4px;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.2s;
}

.text-item:hover {
  background: #f0f0f0;
}

.text-item.active {
  background: #e6f0ff;
}

.text-index {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #409eff;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  flex-shrink: 0;
}

.text-item .el-input {
  flex: 1;
}
</style>