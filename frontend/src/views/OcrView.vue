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
          >
            <div
              style="margin: -8px -20px; padding: 8px 20px;"
              @mouseenter="onImgOptionEnter(idx, $event)"
              @mouseleave="onPreviewLeave"
            >
              图片 {{ idx + 1 }}
            </div>
          </el-option>
        </el-select>
        <el-button type="primary" @click="runPredict" :loading="predictLoading">
          执行 Predict
        </el-button>
      </div>
    </div>

    <div v-if="ocrResults.length === 0" class="empty-state">
      <el-empty description="暂无 OCR 结果，请先执行 OCR 识别" />
    </div>

    <div v-else class="ocr-workspace" ref="workspaceRef">
      <div class="canvas-container" ref="canvasContainer">
        <v-stage
          ref="stageRef"
          :config="stageConfig"
          @mousedown="handleStageMouseDown"
          @mousemove="handleStageMouseMove"
          @mouseup="handleStageMouseUp"
        >
          <v-layer>
            <v-image :config="imageConfig" />

            <v-rect
              v-for="(box, idx) in currentBoxes"
              :key="'box-' + idx"
              :config="getBoxConfig(idx, box)"
            />

            <v-rect
              v-for="(box, idx) in currentBoxes"
              :key="'label-bg-' + idx"
              :config="getLabelBgConfig(idx, box)"
            />

            <v-text
              v-for="(box, idx) in currentBoxes"
              :key="'label-' + idx"
              :config="getLabelConfig(idx, box)"
            />

            <v-circle
              v-for="(box, idx) in currentBoxes"
              :key="'handle-tl-' + idx"
              :config="getHandleConfig(idx, box, 'tl')"
            />

            <v-circle
              v-for="(box, idx) in currentBoxes"
              :key="'handle-br-' + idx"
              :config="getHandleConfig(idx, box, 'br')"
            />
          </v-layer>
        </v-stage>
      </div>

      <div class="text-panel" ref="textPanelRef">
        <h3>文本列表</h3>
        <draggable
          v-model="currentTexts"
          :animation="200"
          :item-key="(_: string, idx: number) => idx"
          ghost-class="text-entry--ghost"
          chosen-class="text-entry--chosen"
          drag-class="text-entry--drag"
          @start="onTextDragStart"
          @end="onTextDragEnd"
          tag="div"
          class="draggable-text-list"
        >
          <template #item="{ element: text, index: idx }">
            <div
              :ref="(el: any) => setTextEntryRef(el, idx)"
              class="text-entry"
              :class="{ active: selectedBoxIdx === idx }"
              @click="selectBox(idx)"
            >
              <div class="text-entry-header">
                <span class="text-entry-id">{{ idx }}</span>
                <span class="drag-handle">⠿</span>
                <el-button
                  type="danger"
                  size="small"
                  circle
                  :icon="Delete"
                  @click.stop="deleteBox(idx)"
                />
              </div>
              <el-input
                v-model="currentTexts[idx]"
                type="textarea"
                :autosize="{ minRows: 1, maxRows: 10 }"
                class="text-entry-input"
                @focus="selectBox(idx)"
                @change="updateText(idx, currentTexts[idx])"
              />
            </div>
          </template>
        </draggable>

        <div class="text-entry text-entry--add" @click="addBox">
          <el-icon :size="22"><Plus /></el-icon>
          <span>添加文本条目</span>
        </div>

        <el-empty v-if="currentTexts.length === 0" description="暂无文本" :image-size="40" />
      </div>
    </div>
  </div>

  <Teleport to="body">
    <div
      v-show="previewVisible"
      class="option-preview-float"
      :style="previewStyle"
    >
      <div style="width: 220px; overflow: hidden; border-radius: 4px; line-height: 0;">
        <img
          :src="`/api/images/serve/${previewIdx}?t=${Date.now()}`"
          style="width: 100%; display: block;"
        />
      </div>
    </div>
  </Teleport>

  <Teleport to="body">
    <div
      v-show="magnifierVisible"
      class="magnifier-float"
      :style="magnifierStyle"
    >
      <canvas
        ref="magnifierCanvasRef"
        :width="MAGNIFIER_SIZE"
        :height="MAGNIFIER_SIZE"
        class="magnifier-canvas"
      />
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Delete, Plus } from '@element-plus/icons-vue'
import draggable from 'vuedraggable'
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
const textPanelRef = ref<HTMLElement | null>(null)
const workspaceRef = ref<HTMLElement | null>(null)

const imageObj = ref<HTMLImageElement | null>(null)
const canvasWidth = ref(900)
const canvasHeight = ref(600)
const imageNaturalWidth = ref(1)
const imageNaturalHeight = ref(1)
const scaleX = ref(1)
const scaleY = ref(1)

const dragState = ref<{ boxIdx: number; corner: 'tl' | 'br' } | null>(null)
const textEntryRefs = ref<Record<number, HTMLElement>>({})
const oldTexts = ref<string[]>([])

const previewVisible = ref(false)
const previewIdx = ref(0)
const previewStyle = ref({ top: '0px', left: '0px' })
let hideTimer: ReturnType<typeof setTimeout> | null = null

const MAGNIFIER_SIZE = 160
const MAGNIFIER_ZOOM = 1
const magnifierVisible = ref(false)
const magnifierStyle = ref({ top: '0px', left: '0px' })
const magnifierCanvasRef = ref<HTMLCanvasElement | null>(null)

function onImgOptionEnter(idx: number, e: MouseEvent) {
  if (hideTimer) { clearTimeout(hideTimer); hideTimer = null }
  previewIdx.value = idx
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
  previewStyle.value = {
    top: rect.top + 'px',
    left: (rect.left - 232) + 'px',
  }
  previewVisible.value = true
}

function onPreviewLeave() {
  hideTimer = setTimeout(() => { previewVisible.value = false }, 100)
}

const HANDLE_RADIUS = 4

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

function toImageX(cx: number) {
  return cx / scaleX.value
}

function toImageY(cy: number) {
  return cy / scaleY.value
}

function getBoxConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0])
  const y = toCanvasY(box[1])
  const w = toCanvasX(box[2]) - x
  const h = toCanvasY(box[3]) - y
  const isSelected = selectedBoxIdx.value === idx
  return {
    x,
    y,
    width: w,
    height: h,
    stroke: isSelected ? '#409eff' : '#00cc66',
    strokeWidth: isSelected ? 2.5 : 1.5,
    fill: isSelected ? 'rgba(64, 158, 255, 0.08)' : 'rgba(0, 204, 102, 0.04)',
    name: 'box-' + idx,
    listening: true,
  }
}

function getLabelBgConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0])
  const y = toCanvasY(box[1])
  const isSelected = selectedBoxIdx.value === idx
  const labelY = Math.max(0, y - 18)
  const numDigits = String(idx).length
  const bgWidth = numDigits * 7 + 6
  return {
    x: x - 2,
    y: labelY,
    width: bgWidth,
    height: 14,
    fill: isSelected ? '#409eff' : '#00cc66',
    cornerRadius: 3,
    name: 'label-bg-' + idx,
    listening: true,
  }
}

function getLabelConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0])
  const y = toCanvasY(box[1])
  const labelY = Math.max(0, y - 18)
  return {
    x: x + 1,
    y: labelY + 1,
    text: `${idx}`,
    fontSize: 12,
    fontStyle: 'bold',
    fill: '#fff',
    name: 'label-' + idx,
    listening: true,
  }
}

function getHandleConfig(idx: number, box: number[], corner: 'tl' | 'br') {
  const isSelected = selectedBoxIdx.value === idx
  const cx = corner === 'tl' ? toCanvasX(box[0]) : toCanvasX(box[2])
  const cy = corner === 'tl' ? toCanvasY(box[1]) : toCanvasY(box[3])
  return {
    x: cx,
    y: cy,
    radius: HANDLE_RADIUS,
    fill: isSelected ? '#409eff' : '#00cc66',
    stroke: '#fff',
    strokeWidth: 2,
    name: 'handle-' + corner + '-' + idx,
    listening: true,
  }
}

function setTextEntryRef(el: any, idx: number) {
  if (el) {
    textEntryRefs.value[idx] = el.$el || el
  }
}

function selectBox(idx: number) {
  selectedBoxIdx.value = idx
  nextTick(() => {
    const entry = textEntryRefs.value[idx]
    if (entry) {
      entry.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  })
}

function handleStageMouseDown(e: any) {
  const name = e.target?.name?.()
  if (!name) {
    selectedBoxIdx.value = null
    return
  }

  if (name.startsWith('handle-tl-')) {
    const idx = parseInt(name.split('-')[2])
    dragState.value = { boxIdx: idx, corner: 'tl' }
    selectBox(idx)
    e.evt.preventDefault()
    return
  }

  if (name.startsWith('handle-br-')) {
    const idx = parseInt(name.split('-')[2])
    dragState.value = { boxIdx: idx, corner: 'br' }
    selectBox(idx)
    e.evt.preventDefault()
    return
  }

  if (name.startsWith('box-')) {
    const idx = parseInt(name.split('-')[1])
    selectBox(idx)
    return
  }

  if (name.startsWith('label-') || name.startsWith('label-bg-')) {
    const idx = parseInt(name.split('-').pop()!)
    selectBox(idx)
    return
  }

  selectedBoxIdx.value = null
}

function handleStageMouseMove(e: any) {
  if (!dragState.value) return

  const stage = stageRef.value?.getStage()
  if (!stage) return

  const pos = stage.getPointerPosition()
  if (!pos) return

  const imgX = toImageX(pos.x)
  const imgY = toImageY(pos.y)

  const { boxIdx, corner } = dragState.value
  const boxes = ocrResults.value[currentImgIdx.value].boxes
  const oldBox = boxes[boxIdx]
  const newBox = [...oldBox]

  if (corner === 'tl') {
    newBox[0] = Math.max(0, Math.min(imgX, oldBox[2] - 10))
    newBox[1] = Math.max(0, Math.min(imgY, oldBox[3] - 10))
  } else {
    newBox[2] = Math.max(oldBox[0] + 10, Math.min(imgX, imageNaturalWidth.value))
    newBox[3] = Math.max(oldBox[1] + 10, Math.min(imgY, imageNaturalHeight.value))
  }

  boxes.splice(boxIdx, 1, newBox)

  updateMagnifier(e, imgX, imgY, corner)
}

function handleStageMouseUp(_e: any) {
  if (!dragState.value) return

  const { boxIdx } = dragState.value
  const box = ocrResults.value[currentImgIdx.value].boxes[boxIdx]

  ocrApi.updateBox(currentImgIdx.value, boxIdx, [...box]).catch(() => {
    ElMessage.error('更新框坐标失败')
  })

  dragState.value = null
  magnifierVisible.value = false
}

function updateMagnifier(e: any, imgX: number, imgY: number, corner: 'tl' | 'br') {
  const canvas = magnifierCanvasRef.value
  if (!canvas || !imageObj.value) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const halfView = (MAGNIFIER_SIZE / MAGNIFIER_ZOOM) / 2
  const sx = Math.max(0, imgX - halfView)
  const sy = Math.max(0, imgY - halfView)
  const sw = Math.min(imageNaturalWidth.value - sx, halfView * 2)
  const sh = Math.min(imageNaturalHeight.value - sy, halfView * 2)

  ctx.clearRect(0, 0, MAGNIFIER_SIZE, MAGNIFIER_SIZE)
  ctx.imageSmoothingEnabled = false

  const dx = (halfView - (imgX - sx)) * MAGNIFIER_ZOOM
  const dy = (halfView - (imgY - sy)) * MAGNIFIER_ZOOM
  ctx.drawImage(imageObj.value, sx, sy, sw, sh, dx, dy, sw * MAGNIFIER_ZOOM, sh * MAGNIFIER_ZOOM)

  const cx = MAGNIFIER_SIZE / 2
  const cy = MAGNIFIER_SIZE / 2
  ctx.strokeStyle = '#ff0000'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(cx, 0)
  ctx.lineTo(cx, MAGNIFIER_SIZE)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(0, cy)
  ctx.lineTo(MAGNIFIER_SIZE, cy)
  ctx.stroke()

  magnifierVisible.value = true

  const evt = e.evt as MouseEvent
  if (corner === 'tl') {
    magnifierStyle.value = {
      top: (evt.clientY - MAGNIFIER_SIZE - 16) + 'px',
      left: (evt.clientX - MAGNIFIER_SIZE - 16) + 'px',
    }
  } else {
    magnifierStyle.value = {
      top: (evt.clientY + 16) + 'px',
      left: (evt.clientX + 16) + 'px',
    }
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
    if (selectedBoxIdx.value === idx) {
      selectedBoxIdx.value = null
    } else if (selectedBoxIdx.value !== null && selectedBoxIdx.value > idx) {
      selectedBoxIdx.value--
    }
    ElMessage.success('已删除')
  } catch {
    ElMessage.error('删除失败')
  }
}

function onTextDragStart() {
  oldTexts.value = [...currentTexts.value]
}

async function onTextDragEnd() {
  if (oldTexts.value.length === 0) return

  const newOrder = currentTexts.value.map((t) => oldTexts.value.indexOf(t))
  const isChanged = newOrder.some((val, idx) => val !== idx)
  if (!isChanged) {
    oldTexts.value = []
    return
  }

  const boxes = ocrResults.value[currentImgIdx.value].boxes
  const oldBoxes = [...boxes]
  const newBoxes = newOrder.map((i) => oldBoxes[i])
  ocrResults.value[currentImgIdx.value].boxes = newBoxes

  try {
    await ocrApi.reorder(currentImgIdx.value, newOrder)
  } catch {
    currentTexts.value = oldTexts.value
    ocrResults.value[currentImgIdx.value].boxes = oldBoxes
    ElMessage.error('排序失败')
  }
  oldTexts.value = []
}

async function addBox() {
  const imgW = imageNaturalWidth.value
  const imgH = imageNaturalHeight.value
  const boxW = Math.round(imgW * 0.15)
  const boxH = Math.round(imgH * 0.06)
  const cx = Math.round(imgW / 2 - boxW / 2)
  const cy = Math.round(imgH / 2 - boxH / 2)
  const newBox = [cx, cy, cx + boxW, cy + boxH]

  try {
    const res = await ocrApi.addBox(currentImgIdx.value, newBox, '')
    const newIdx = res.data.idx
    ocrResults.value[currentImgIdx.value].boxes.push(newBox)
    ocrResults.value[currentImgIdx.value].texts.push('')
    await nextTick()
    selectBox(newIdx)
  } catch {
    ElMessage.error('添加失败')
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

    const workspaceWidth = workspaceRef.value?.clientWidth || 1200
    const maxHeight = 700
    const textPanelWidth = 320
    const gap = 16
    let w = workspaceWidth - textPanelWidth - gap
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
  dragState.value = null
  await nextTick()
  await loadImage()
})

onMounted(loadOcrResults)
</script>

<style scoped>
.ocr-page {
  max-width: 1500px;
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
  align-items: flex-start;
}

.canvas-container {
  flex-shrink: 0;
  overflow: hidden;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background: #f5f5f5;
  cursor: default;
}

.text-panel {
  width: 320px;
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
  position: sticky;
  top: 0;
  background: #fff;
  padding-bottom: 8px;
  z-index: 1;
}

.text-entry {
  margin-bottom: 10px;
  padding: 8px;
  border: 1px solid #e8e8e8;
  border-radius: 6px;
  cursor: grab;
  transition: all 0.2s;
  background: #fff;
}

.text-entry:hover {
  border-color: #c0c0c0;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
}

.text-entry.active {
  border-color: #409eff;
  background: #ecf5ff;
  box-shadow: 0 1px 6px rgba(64, 158, 255, 0.15);
}

.text-entry-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.drag-handle {
  cursor: grab;
  color: #999;
  font-size: 16px;
  line-height: 1;
  user-select: none;
  opacity: 0;
  transition: opacity 0.2s;
}

.text-entry:hover .drag-handle {
  opacity: 1;
}

.drag-handle:active {
  cursor: grabbing;
}

.text-entry--ghost {
  opacity: 0.4;
  background: #f0f0f0;
  border: 2px dashed #c0c0c0;
}

.text-entry--chosen {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.text-entry--drag {
  opacity: 0.8;
  transform: rotate(2deg);
}

.text-entry--add {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 2px dashed #d0d0d0;
  color: #999;
  padding: 14px 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.text-entry--add:hover {
  border-color: #409eff;
  color: #409eff;
  background: #ecf5ff;
}

.text-entry-id {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: #409eff;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  flex-shrink: 0;
}

.text-entry.active .text-entry-id {
  background: #337ecc;
  box-shadow: 0 0 0 3px rgba(64, 158, 255, 0.25);
}

.text-entry-input {
  width: 100%;
}

.text-entry-input :deep(.el-textarea__inner) {
  font-size: 13px;
  line-height: 1.5;
  resize: none;
  border-color: transparent;
  background: transparent;
  padding: 4px 6px;
  box-shadow: none;
}

.text-entry.active .text-entry-input :deep(.el-textarea__inner) {
  background: #fff;
  border-color: #d9d9d9;
}

.text-entry-input :deep(.el-textarea__inner):focus {
  border-color: #409eff;
  background: #fff;
}
</style>

<style>
.option-preview-float {
  position: fixed;
  z-index: 10000;
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
  background: #fff;
  padding: 4px;
  pointer-events: none;
}

.magnifier-float {
  position: fixed;
  z-index: 10001;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
  background: #fff;
  padding: 3px;
  pointer-events: none;
}

.magnifier-canvas {
  display: block;
  border-radius: 4px;
}
</style>