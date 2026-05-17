<template>
  <div class="predict-page">
    <div class="page-header">
      <h2>Predict 预测结果</h2>
      <div class="header-actions">
        <el-select v-model="currentImgIdx" placeholder="选择图片" style="width: 200px">
          <el-option v-for="(_, idx) in results" :key="idx" :label="`图片 ${idx + 1}`" :value="idx">
            <div
              style="margin: -8px -20px; padding: 8px 20px;"
              @mouseenter="onImgOptionEnter(idx, $event)"
              @mouseleave="onPreviewLeave"
            >
              图片 {{ idx + 1 }}
            </div>
          </el-option>
        </el-select>
        <el-button v-if="selectedCharIdx !== null" type="danger" @click="deleteSelectedCharacter">
          删除人物框
        </el-button>
        <el-button v-else @click="addCharacterMode = !addCharacterMode" :type="addCharacterMode ? 'warning' : 'default'">
          {{ addCharacterMode ? '取消添加' : '添加人物框' }}
        </el-button>
        <el-button type="primary" @click="runGrounding" :loading="groundingLoading">
          执行 Grounding
        </el-button>
      </div>
    </div>

    <div v-if="results.length === 0" class="empty-state">
      <el-empty description="暂无 Predict 结果，请先执行 Predict" />
    </div>

    <div v-else class="predict-workspace" ref="workspaceRef">
      <div class="canvas-container" ref="canvasContainer">
        <v-stage ref="stageRef" :config="stageConfig"
          @mousedown="handleStageMouseDown"
          @mousemove="handleStageMouseMove"
          @mouseup="handleStageMouseUp"
        >
          <v-layer>
            <v-image :config="imageConfig" />

            <v-line v-for="(assoc, idx) in currentAssociations" :key="'cline-' + idx"
              :config="getCharLineConfig(assoc)" />

            <v-rect v-for="(box, idx) in currentCharacters" :key="'cbox-' + idx"
              :config="getCharBoxConfig(idx, box)" />

            <v-rect v-for="(box, idx) in currentCharacters" :key="'clabel-bg-' + idx"
              :config="getCharLabelBgConfig(idx, box)" />

            <v-text v-for="(box, idx) in currentCharacters" :key="'clabel-' + idx"
              :config="getCharLabelConfig(idx, box)" />

            <v-rect v-for="(box, idx) in currentTextBoxes" :key="'tbox-' + idx"
              :config="getTextBoxConfig(idx, box)" />

            <template v-for="(box, idx) in currentCharacters" :key="'chandle-tl-' + idx">
              <v-circle v-if="selectedCharIdx === null && highlightedGlobalId === null" :config="getCharHandleConfig(idx, box, 'tl')" />
            </template>

            <template v-for="(box, idx) in currentCharacters" :key="'chandle-br-' + idx">
              <v-circle v-if="selectedCharIdx === null && highlightedGlobalId === null" :config="getCharHandleConfig(idx, box, 'br')" />
            </template>
          </v-layer>
        </v-stage>
      </div>

      <div class="side-panels">
        <div class="global-char-panel" ref="globalCharPanelRef">
          <h3>全局角色库</h3>
          <div
            v-for="entry in sortedGlobalCharLibrary"
            :key="entry.global_id"
            :ref="(el: any) => setGlobalCharRef(el, entry.global_id)"
            class="global-char-item"
            :class="{ active: highlightedGlobalId === entry.global_id }"
            @click="selectGlobalChar(entry.global_id)"
          >
            <div class="global-char-header">
              <span class="global-char-id">{{ entry.global_id }}</span>
              <el-button
                type="danger"
                size="small"
                circle
                :icon="Delete"
                @click.stop="deleteGlobalCharEntry(entry.global_id)"
              />
            </div>
            <div @click.stop>
              <el-input
                v-model="charNameMap[entry.global_id]"
                type="text"
                size="small"
                placeholder="角色名"
                class="global-char-input"
                @focus="selectGlobalChar(entry.global_id)"
                @change="updateCharName(entry.global_id, charNameMap[entry.global_id])"
              />
            </div>
          </div>

          <div class="global-char-item global-char-item--add" @click="addGlobalCharEntry">
            <el-icon :size="22"><Plus /></el-icon>
            <span>添加角色条目</span>
          </div>

          <el-empty v-if="!globalCharLibrary.length" description="暂无全局角色" :image-size="40" />
        </div>
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
import { predictApi, groundingApi, characterApi } from '../api/endpoints'

const router = useRouter()

const COLORS = ['#FF6B6B', '#3E7BFF', '#FF9E4D', '#C77DFF', '#FF4E9F', '#5E5CFF', '#FFBD7A', '#D96EFF', '#FF4D7E', '#4A8CFF', '#E07BFF', '#FF6B9D']

function getColor(id: number) {
  return COLORS[id % COLORS.length]
}

const results = ref<any[]>([])
const currentImgIdx = ref(0)
const selectedCharIdx = ref<number | null>(null)
const highlightedGlobalId = ref<number | null>(null)
const addCharacterMode = ref(false)
const groundingLoading = ref(false)
const charNameMap = ref<Record<number, string>>({})
const globalCharLibrary = ref<any[]>([])

const sortedGlobalCharLibrary = computed(() => {
  return [...globalCharLibrary.value].sort((a, b) => a.global_id - b.global_id)
})

const stageRef = ref<any>(null)
const canvasContainer = ref<HTMLElement | null>(null)
const workspaceRef = ref<HTMLElement | null>(null)
const globalCharPanelRef = ref<HTMLElement | null>(null)
const globalCharRefs = ref<Record<number, HTMLElement>>({})
const imageObj = ref<HTMLImageElement | null>(null)
const canvasWidth = ref(900)
const canvasHeight = ref(600)
const scaleX = ref(1)
const scaleY = ref(1)

const charDragState = ref<{ boxIdx: number; corner: 'tl' | 'br' } | null>(null)

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

const currentResult = computed(() => {
  if (currentImgIdx.value >= results.value.length) return {}
  return results.value[currentImgIdx.value] || {}
})

const currentTextBoxes = computed(() => currentResult.value.texts || [])
const currentOcrTexts = computed(() => currentResult.value.ocr_texts || [])
const currentCharacters = computed(() => currentResult.value.characters || [])
const currentGlobalIds = computed(() => currentResult.value.global_character_ids || [])
const currentAssociations = computed(() => currentResult.value.text_character_associations || [])

const stageConfig = computed(() => ({ width: canvasWidth.value, height: canvasHeight.value }))
const imageConfig = computed(() => ({ image: imageObj.value, width: canvasWidth.value, height: canvasHeight.value }))

function toCanvasX(x: number) { return x * scaleX.value }
function toCanvasY(y: number) { return y * scaleY.value }
function toImageX(cx: number) { return cx / scaleX.value }
function toImageY(cy: number) { return cy / scaleY.value }

function getTextBoxConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0])
  const y = toCanvasY(box[1])
  const w = toCanvasX(box[2]) - x
  const h = toCanvasY(box[3]) - y
  const isCheckMode = selectedCharIdx.value !== null
  const isGlobalMode = selectedCharIdx.value === null && highlightedGlobalId.value !== null

  const associatedChars = currentAssociations.value
    .filter(([t, c]: [number, number]) => t === idx)
    .map(([t, c]: [number, number]) => c)

  let fill: string
  let stroke: string
  let strokeWidth: number
  let opacity: number

  if (isCheckMode) {
    const isAssociated = isTextAssociated(idx, selectedCharIdx.value!)
    if (isAssociated) {
      const color = getColor(currentGlobalIds.value[selectedCharIdx.value!])
      fill = color + '30'
      stroke = color
      strokeWidth = 3
      opacity = 1
    } else {
      fill = 'rgba(128,128,128,0.35)'
      stroke = '#999'
      strokeWidth = 1.5
      opacity = 0.4
    }
  } else if (isGlobalMode) {
    fill = 'rgba(128,128,128,0.35)'
    stroke = '#999'
    strokeWidth = 1.5
    opacity = 0.4
  } else if (associatedChars.length === 1) {
    const charIdx = associatedChars[0]
    const gid = currentGlobalIds.value[charIdx] ?? charIdx
    const color = getColor(gid)
    fill = color + '20'
    stroke = color
    strokeWidth = 2
    opacity = 1
  } else {
    fill = 'rgba(0,204,102,0.04)'
    stroke = '#00cc66'
    strokeWidth = 1.5
    opacity = 1
  }

  return {
    x, y, width: w, height: h,
    fill, stroke, strokeWidth, opacity,
    name: 'tbox-' + idx,
    listening: true,
  }
}

function isTextAssociated(textIdx: number, charIdx: number): boolean {
  return currentAssociations.value.some(
    ([t, c]: [number, number]) => t === textIdx && c === charIdx
  )
}

function getCharBoxConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0])
  const y = toCanvasY(box[1])
  const gid = currentGlobalIds.value[idx] ?? idx
  const isSelected = selectedCharIdx.value === idx
  const isHighlighted = selectedCharIdx.value === null && highlightedGlobalId.value !== null && highlightedGlobalId.value === gid
  const isCheckMode = selectedCharIdx.value !== null
  const isGlobalMode = selectedCharIdx.value === null && highlightedGlobalId.value !== null
  const shouldDim = (isCheckMode && !isSelected) || (isGlobalMode && !isHighlighted)
  return {
    x, y,
    width: toCanvasX(box[2]) - x,
    height: toCanvasY(box[3]) - y,
    stroke: getColor(gid),
    strokeWidth: (isSelected || isHighlighted) ? 3 : 2,
    fill: (isSelected || isHighlighted) ? getColor(gid) + '20' : 'rgba(0,0,0,0)',
    opacity: shouldDim ? 0.15 : 1,
    name: 'cbox-' + idx,
  }
}

function getCharLabelBgConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0])
  const y = toCanvasY(box[1])
  const gid = currentGlobalIds.value[idx] ?? idx
  const isSelected = selectedCharIdx.value === idx
  const isHighlighted = selectedCharIdx.value === null && highlightedGlobalId.value !== null && highlightedGlobalId.value === gid
  const isCheckMode = selectedCharIdx.value !== null
  const isGlobalMode = selectedCharIdx.value === null && highlightedGlobalId.value !== null
  const shouldDim = (isCheckMode && !isSelected) || (isGlobalMode && !isHighlighted)
  const labelY = Math.max(0, y - 18)
  const numDigits = String(gid).length
  const bgWidth = numDigits * 7 + 6
  return {
    x: x - 2,
    y: labelY,
    width: bgWidth,
    height: 14,
    fill: getColor(gid),
    cornerRadius: 3,
    opacity: shouldDim ? 0.15 : 1,
    name: 'clabel-bg-' + idx,
    listening: true,
  }
}

function getCharLabelConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0])
  const y = toCanvasY(box[1])
  const gid = currentGlobalIds.value[idx] ?? idx
  const isSelected = selectedCharIdx.value === idx
  const isHighlighted = selectedCharIdx.value === null && highlightedGlobalId.value !== null && highlightedGlobalId.value === gid
  const isCheckMode = selectedCharIdx.value !== null
  const isGlobalMode = selectedCharIdx.value === null && highlightedGlobalId.value !== null
  const shouldDim = (isCheckMode && !isSelected) || (isGlobalMode && !isHighlighted)
  const labelY = Math.max(0, y - 18)
  return {
    x: x + 1,
    y: labelY + 1,
    text: `${gid}`,
    fontSize: 12,
    fontStyle: 'bold',
    fill: '#fff',
    opacity: shouldDim ? 0.15 : 1,
    name: 'clabel-' + idx,
    listening: true,
  }
}

const CHAR_HANDLE_RADIUS = 4

function isCharActive(idx: number): boolean {
  if (selectedCharIdx.value === idx) return true
  const gid = currentGlobalIds.value[idx] ?? idx
  return selectedCharIdx.value === null && highlightedGlobalId.value !== null && highlightedGlobalId.value === gid
}

function getCharHandleConfig(idx: number, box: number[], corner: 'tl' | 'br') {
  const gid = currentGlobalIds.value[idx] ?? idx
  const isSelected = selectedCharIdx.value === idx
  const isHighlighted = selectedCharIdx.value === null && highlightedGlobalId.value !== null && highlightedGlobalId.value === gid
  const isCheckMode = selectedCharIdx.value !== null
  const isGlobalMode = selectedCharIdx.value === null && highlightedGlobalId.value !== null
  const shouldDim = (isCheckMode && !isSelected) || (isGlobalMode && !isHighlighted)
  const cx = corner === 'tl' ? toCanvasX(box[0]) : toCanvasX(box[2])
  const cy = corner === 'tl' ? toCanvasY(box[1]) : toCanvasY(box[3])
  return {
    x: cx,
    y: cy,
    radius: CHAR_HANDLE_RADIUS,
    fill: getColor(gid),
    stroke: '#fff',
    strokeWidth: 2,
    opacity: shouldDim ? 0.15 : 1,
    name: 'chandle-' + corner + '-' + idx,
    listening: !isCharActive(idx),
  }
}

function getCharLineConfig(assoc: number[]) {
  const tIdx = assoc[0]
  const cIdx = assoc[1]
  const tBox = currentTextBoxes.value[tIdx]
  const cBox = currentCharacters.value[cIdx]
  if (!tBox || !cBox) return { points: [0, 0, 0, 0], stroke: '#999', strokeWidth: 1, dash: [4, 4] }
  const tx = toCanvasX((tBox[0] + tBox[2]) / 2)
  const ty = toCanvasY((tBox[1] + tBox[3]) / 2)
  const cx = toCanvasX((cBox[0] + cBox[2]) / 2)
  const cy = toCanvasY((cBox[1] + cBox[3]) / 2)
  const gid = currentGlobalIds.value[cIdx] ?? cIdx
  const isSelected = selectedCharIdx.value === cIdx
  const isCheckMode = selectedCharIdx.value !== null
  const isGlobalMode = selectedCharIdx.value === null && highlightedGlobalId.value !== null
  const shouldDim = (isCheckMode && !isSelected) || isGlobalMode
  return {
    points: [tx, ty, cx, cy],
    stroke: getColor(gid),
    strokeWidth: isSelected ? 3 : 1.5,
    dash: [6, 4],
    opacity: shouldDim ? 0.15 : 1,
    name: 'cline-' + tIdx + '-' + cIdx,
  }
}

function setGlobalCharRef(el: any, globalId: number) {
  if (el) {
    globalCharRefs.value[globalId] = el.$el || el
  }
}

function selectCharacter(idx: number) {
  if (selectedCharIdx.value === idx) {
    selectedCharIdx.value = null
    highlightedGlobalId.value = null
  } else {
    selectedCharIdx.value = idx
    const gid = currentGlobalIds.value[idx]
    highlightedGlobalId.value = gid ?? idx
    scrollToGlobalChar(gid ?? idx)
  }
}

function selectGlobalChar(globalId: number) {
  if (selectedCharIdx.value !== null) {
    const charIdx = selectedCharIdx.value
    const oldGid = currentGlobalIds.value[charIdx] ?? charIdx
    if (oldGid !== globalId) {
      currentGlobalIds.value[charIdx] = globalId
      highlightedGlobalId.value = globalId
      characterApi.updateGlobalId(currentImgIdx.value, charIdx, globalId).catch(() => {})
    }
    scrollToGlobalChar(globalId)
    return
  }

  if (highlightedGlobalId.value === globalId) {
    highlightedGlobalId.value = null
  } else {
    highlightedGlobalId.value = globalId
  }
  scrollToGlobalChar(globalId)
}

function scrollToGlobalChar(globalId: number) {
  nextTick(() => {
    const el = globalCharRefs.value[globalId]
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  })
}

function toggleTextAssociation(textIdx: number) {
  if (selectedCharIdx.value === null) return
  const charIdx = selectedCharIdx.value
  const currentAssocs: [number, number][] = [...(results.value[currentImgIdx.value].text_character_associations || [])]

  const existingIdx = currentAssocs.findIndex(
    ([t, c]: [number, number]) => t === textIdx && c === charIdx
  )

  if (existingIdx >= 0) {
    currentAssocs.splice(existingIdx, 1)
    predictApi.updateAssociation(currentImgIdx.value, textIdx, null).catch(() => {})
  } else {
    currentAssocs.push([textIdx, charIdx])
    predictApi.updateAssociation(currentImgIdx.value, textIdx, charIdx).catch(() => {})
  }

  results.value[currentImgIdx.value].text_character_associations = currentAssocs
}

function deleteSelectedCharacter() {
  if (selectedCharIdx.value === null) return
  const charIdx = selectedCharIdx.value
  predictApi.deleteCharacter(currentImgIdx.value, charIdx).then(() => {
    results.value[currentImgIdx.value].characters.splice(charIdx, 1)
    const gids = results.value[currentImgIdx.value].global_character_ids
    if (charIdx < gids.length) {
      gids.splice(charIdx, 1)
    }
    const assocs = results.value[currentImgIdx.value].text_character_associations || []
    const newAssocs: [number, number][] = []
    for (const [t, c] of assocs) {
      if (c === charIdx) continue
      newAssocs.push([t, c > charIdx ? c - 1 : c])
    }
    results.value[currentImgIdx.value].text_character_associations = newAssocs
    selectedCharIdx.value = null
    highlightedGlobalId.value = null
    ElMessage.success('已删除人物框')
  }).catch(() => {
    ElMessage.error('删除人物框失败')
  })
}

function deleteGlobalCharEntry(globalId: number) {
  characterApi.deleteFromLibrary(globalId).then(() => {
    globalCharLibrary.value = globalCharLibrary.value.filter(
      (entry: any) => entry.global_id !== globalId
    )
    delete charNameMap.value[globalId]
    if (highlightedGlobalId.value === globalId) {
      highlightedGlobalId.value = null
    }
    ElMessage.success('已从全局角色库中删除')
  }).catch((e: any) => {
    ElMessage.error(e.response?.data?.detail || '删除失败')
  })
}

function addGlobalCharEntry() {
  characterApi.addToLibrary().then((res) => {
    const newGid = res.data.global_id
    globalCharLibrary.value.push({ global_id: newGid })
    nextTick(() => {
      scrollToGlobalChar(newGid)
    })
    ElMessage.success('已添加角色条目')
  }).catch((e: any) => {
    ElMessage.error(e.response?.data?.detail || '添加失败')
  })
}

function handleStageMouseDown(e: any) {
  const name = e.target?.name?.()

  if (addCharacterMode.value) {
    const stage = stageRef.value?.getStage()
    if (!stage) return
    const pos = stage.getPointerPosition()
    if (!pos) return
    const imgX = pos.x / scaleX.value
    const imgY = pos.y / scaleY.value
    const box = [imgX, imgY, imgX + 50, imgY + 50]
    const wasLibraryEmpty = globalCharLibrary.value.length === 0
    predictApi.addCharacter(currentImgIdx.value, box).then((res) => {
      results.value[currentImgIdx.value].characters.push(box)
      const assignedGid = res.data.global_id
      results.value[currentImgIdx.value].global_character_ids.push(assignedGid)
      if (wasLibraryEmpty) {
        loadCharLibrary().then(() => {
          nextTick(() => {
            scrollToGlobalChar(assignedGid)
          })
        })
      }
      ElMessage.success('已添加人物框')
    }).catch(() => {
      ElMessage.error('添加人物框失败')
    })
    addCharacterMode.value = false
    return
  }

  if (name && name.startsWith('chandle-tl-')) {
    const idx = parseInt(name.split('-')[2])
    charDragState.value = { boxIdx: idx, corner: 'tl' }
    e.evt.preventDefault()
    return
  }

  if (name && name.startsWith('chandle-br-')) {
    const idx = parseInt(name.split('-')[2])
    charDragState.value = { boxIdx: idx, corner: 'br' }
    e.evt.preventDefault()
    return
  }

  if (name && name.startsWith('tbox-')) {
    const idx = parseInt(name.split('-')[1])
    if (selectedCharIdx.value !== null) {
      toggleTextAssociation(idx)
    }
    return
  }

  if (name && name.startsWith('cbox-')) {
    const idx = parseInt(name.split('-')[1])
    selectCharacter(idx)
    return
  }

  if (name && (name.startsWith('clabel-') || name.startsWith('clabel-bg-'))) {
    const idx = parseInt(name.split('-').pop()!)
    selectCharacter(idx)
    return
  }

  selectedCharIdx.value = null
  highlightedGlobalId.value = null
}

function handleStageMouseMove(e: any) {
  if (!charDragState.value) return

  const stage = stageRef.value?.getStage()
  if (!stage) return

  const pos = stage.getPointerPosition()
  if (!pos) return

  const imgX = toImageX(pos.x)
  const imgY = toImageY(pos.y)

  const { boxIdx, corner } = charDragState.value
  const chars = results.value[currentImgIdx.value].characters
  const oldBox = chars[boxIdx]
  const newBox = [...oldBox]

  if (corner === 'tl') {
    newBox[0] = Math.max(0, Math.min(imgX, oldBox[2] - 10))
    newBox[1] = Math.max(0, Math.min(imgY, oldBox[3] - 10))
  } else {
    newBox[2] = Math.max(oldBox[0] + 10, imgX)
    newBox[3] = Math.max(oldBox[1] + 10, imgY)
  }

  chars.splice(boxIdx, 1, newBox)

  updateMagnifier(e, imgX, imgY, corner)
}

function handleStageMouseUp(_e: any) {
  if (!charDragState.value) return

  const { boxIdx } = charDragState.value
  const box = results.value[currentImgIdx.value].characters[boxIdx]

  predictApi.updateCharacterBox(currentImgIdx.value, boxIdx, [...box]).catch(() => {
    ElMessage.error('更新角色框坐标失败')
  })

  charDragState.value = null
  magnifierVisible.value = false
}

function updateMagnifier(e: any, imgX: number, imgY: number, corner: 'tl' | 'br') {
  const canvas = magnifierCanvasRef.value
  if (!canvas || !imageObj.value) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const imgW = imageObj.value.naturalWidth
  const imgH = imageObj.value.naturalHeight
  const halfView = (MAGNIFIER_SIZE / MAGNIFIER_ZOOM) / 2
  const sx = Math.max(0, imgX - halfView)
  const sy = Math.max(0, imgY - halfView)
  const sw = Math.min(imgW - sx, halfView * 2)
  const sh = Math.min(imgH - sy, halfView * 2)

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

async function updateCharName(globalId: number, name: string) {
  try {
    await characterApi.updateName(globalId, name)
  } catch { ElMessage.error('更新角色名失败') }
}

async function loadImage() {
  if (results.value.length === 0) return
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.src = `/api/images/serve/${currentImgIdx.value}?t=${Date.now()}`
  img.onload = () => {
    imageObj.value = img
    const workspaceWidth = workspaceRef.value?.clientWidth || 1200
    const maxHeight = 700
    const sidePanelWidth = 300
    const gap = 16
    let w = workspaceWidth - sidePanelWidth - gap
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

async function loadResults() {
  try {
    const res = await predictApi.results()
    results.value = res.data.results || []
    if (results.value.length > 0) {
      await nextTick()
      await loadImage()
    }
  } catch { results.value = [] }
}

async function loadCharLibrary() {
  try {
    const res = await characterApi.library()
    globalCharLibrary.value = res.data.global_character_library || []
    charNameMap.value = res.data.character_name_map || {}
  } catch {}
}

async function runGrounding() {
  groundingLoading.value = true
  try {
    await groundingApi.run()
    ElMessage.success('Grounding 完成')
    router.push('/grounding')
  } catch (e: any) {
    ElMessage.error('Grounding 失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    groundingLoading.value = false
  }
}

watch(currentImgIdx, async () => {
  selectedCharIdx.value = null
  highlightedGlobalId.value = null
  await nextTick()
  await loadImage()
})

onMounted(() => { loadResults(); loadCharLibrary() })
</script>

<style scoped>
.predict-page { max-width: 1500px; margin: 0 auto; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.page-header h2 { margin: 0; }
.header-actions { display: flex; gap: 12px; align-items: center; }
.empty-state { margin-top: 60px; }

.predict-workspace { display: flex; gap: 16px; align-items: flex-start; }

.canvas-container {
  flex-shrink: 0;
  overflow: hidden;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background: #f5f5f5;
  cursor: default;
}

.side-panels { width: 300px; flex-shrink: 0; display: flex; flex-direction: column; gap: 12px; }

.global-char-panel {
  border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px;
  max-height: 700px; overflow-y: auto;
}
.global-char-panel h3 {
  margin: 0 0 12px 0;
  font-size: 15px;
  position: sticky;
  top: 0;
  background: #fff;
  padding-bottom: 8px;
  z-index: 1;
}

.global-char-item {
  margin-bottom: 10px;
  padding: 8px;
  border: 1px solid #e8e8e8;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  background: #fff;
}
.global-char-item:hover {
  border-color: #c0c0c0;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
}
.global-char-item.active {
  border-color: #409eff;
  background: #ecf5ff;
  box-shadow: 0 1px 6px rgba(64, 158, 255, 0.15);
}

.global-char-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.global-char-id {
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
.global-char-item.active .global-char-id {
  background: #337ecc;
  box-shadow: 0 0 0 3px rgba(64, 158, 255, 0.25);
}

.global-char-input {
  width: 100%;
}
.global-char-input :deep(.el-input__wrapper) {
  border-color: transparent;
  background: transparent;
  box-shadow: none;
  padding: 4px 6px;
}
.global-char-input :deep(.el-input__inner) {
  font-size: 13px;
  line-height: 1.5;
  border-color: transparent;
  background: transparent;
  padding: 0;
  box-shadow: none;
}
.global-char-item.active .global-char-input :deep(.el-input__wrapper) {
  background: #fff;
  border-color: #d9d9d9;
}
.global-char-item.active .global-char-input :deep(.el-input__inner) {
  background: #fff;
}
.global-char-input :deep(.el-input__wrapper):focus,
.global-char-input :deep(.el-input__wrapper.is-focus) {
  border-color: #409eff;
  background: #fff;
}
.global-char-input :deep(.el-input__inner):focus {
  background: #fff;
}

.global-char-item--add {
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
.global-char-item--add:hover {
  border-color: #409eff;
  color: #409eff;
  background: #ecf5ff;
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