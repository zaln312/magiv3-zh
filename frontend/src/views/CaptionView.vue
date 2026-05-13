<template>
  <div class="caption-page">
    <div class="page-header">
      <h2>Grounded Caption</h2>
      <div class="header-actions">
        <el-select v-model="currentImgIdx" placeholder="选择图片" style="width: 200px">
          <el-option v-for="(_, idx) in results" :key="idx" :label="`图片 ${idx + 1}`" :value="idx" />
        </el-select>
        <el-button type="primary" @click="runGrounding" :loading="groundingLoading">
          执行 Grounding
        </el-button>
      </div>
    </div>

    <div v-if="results.length === 0" class="empty-state">
      <el-empty description="暂无 Caption 结果，请先执行 Caption" />
    </div>

    <div v-else class="caption-workspace">
      <div class="canvas-container" ref="canvasContainer">
        <v-stage ref="stageRef" :config="stageConfig">
          <v-layer>
            <v-image :config="imageConfig" />

            <v-rect v-for="(box, idx) in currentTextBoxes" :key="'tbox-' + idx"
              :config="getTextBoxConfig(idx, box)" />

            <v-rect v-for="(box, idx) in currentCharacters" :key="'cbox-' + idx"
              :config="getCharBoxConfig(idx, box)" />

            <v-text v-for="(box, idx) in currentCharacters" :key="'clabel-' + idx"
              :config="getCharLabelConfig(idx, box)" />

            <v-line v-for="(assoc, idx) in currentAssociations" :key="'cline-' + idx"
              :config="getCharLineConfig(assoc)" />
          </v-layer>
        </v-stage>
      </div>

      <div class="caption-panel">
        <h3>Caption 文本</h3>
        <div v-if="currentCaption" class="caption-text">
          <el-input
            v-model="currentCaption"
            type="textarea"
            :rows="12"
            resize="none"
            @change="updateCaption"
          />
        </div>
        <el-empty v-else description="暂无 Caption" :image-size="40" />

        <h3 style="margin-top: 16px">Panel Script</h3>
        <div v-if="currentPanelScript" class="panel-script">
          <el-input
            v-model="currentPanelScript"
            type="textarea"
            :rows="8"
            resize="none"
            @change="updatePanelScript"
          />
        </div>
        <el-empty v-else description="暂无 Panel Script" :image-size="40" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { captionApi, groundingApi } from '../api/endpoints'

const router = useRouter()

const COLORS = ['#E00000', '#00CE00', '#0000FF', '#DBDB06', '#DD00DD', '#00E0E0',
  '#FFA500', '#800080', '#008000', '#000080', '#A52A2A', '#D8A4AD']

function getColor(id: number) { return COLORS[id % COLORS.length] }

const results = ref<any[]>([])
const currentImgIdx = ref(0)
const groundingLoading = ref(false)

const canvasContainer = ref<HTMLElement | null>(null)
const imageObj = ref<HTMLImageElement | null>(null)
const canvasWidth = ref(900)
const canvasHeight = ref(600)
const scaleX = ref(1)
const scaleY = ref(1)

const currentResult = computed(() => results.value[currentImgIdx.value] || {})
const currentCaption = computed({
  get: () => currentResult.value.caption || '',
  set: (val) => { if (results.value[currentImgIdx.value]) results.value[currentImgIdx.value].caption = val },
})
const currentPanelScript = computed({
  get: () => currentResult.value.panel_script || '',
  set: (val) => { if (results.value[currentImgIdx.value]) results.value[currentImgIdx.value].panel_script = val },
})
const currentTextBoxes = computed(() => currentResult.value.texts || [])
const currentCharacters = computed(() => currentResult.value.characters || [])
const currentGlobalIds = computed(() => currentResult.value.global_character_ids || [])
const currentAssociations = computed(() => currentResult.value.text_character_associations || [])

const stageConfig = computed(() => ({ width: canvasWidth.value, height: canvasHeight.value }))
const imageConfig = computed(() => ({ image: imageObj.value, width: canvasWidth.value, height: canvasHeight.value }))

function toCanvasX(x: number) { return x * scaleX.value }
function toCanvasY(y: number) { return y * scaleY.value }

function getTextBoxConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0]), y = toCanvasY(box[1])
  return { x, y, width: toCanvasX(box[2]) - x, height: toCanvasY(box[3]) - y, stroke: '#00ff00', strokeWidth: 1, fill: 'rgba(0,255,0,0.03)', name: 'tbox-' + idx }
}

function getCharBoxConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0]), y = toCanvasY(box[1])
  const gid = currentGlobalIds.value[idx] ?? idx
  return { x, y, width: toCanvasX(box[2]) - x, height: toCanvasY(box[3]) - y, stroke: getColor(gid), strokeWidth: 2, fill: 'rgba(0,0,0,0)', name: 'cbox-' + idx }
}

function getCharLabelConfig(idx: number, box: number[]) {
  const gid = currentGlobalIds.value[idx] ?? idx
  return { x: toCanvasX(box[0]), y: Math.max(0, toCanvasY(box[1]) - 18), text: `${gid}`, fontSize: 14, fill: '#fff', name: 'clabel-' + idx }
}

function getCharLineConfig(assoc: number[]) {
  const tIdx = assoc[0], cIdx = assoc[1]
  const tBox = currentTextBoxes.value[tIdx]
  const cBox = currentCharacters.value[cIdx]
  if (!tBox || !cBox) return { points: [0, 0, 0, 0], stroke: '#999', strokeWidth: 1, dash: [4, 4] }
  const gid = currentGlobalIds.value[cIdx] ?? cIdx
  return { points: [toCanvasX((tBox[0] + tBox[2]) / 2), toCanvasY((tBox[1] + tBox[3]) / 2), toCanvasX((cBox[0] + cBox[2]) / 2), toCanvasY((cBox[1] + cBox[3]) / 2)], stroke: getColor(gid), strokeWidth: 1.5, dash: [6, 4], name: 'cline-' + tIdx + '-' + cIdx }
}

async function updateCaption() {
  try { await captionApi.updateCaption(currentImgIdx.value, currentCaption.value) } catch { ElMessage.error('更新失败') }
}

async function updatePanelScript() {
  try { await captionApi.updatePanelScript(currentImgIdx.value, currentPanelScript.value) } catch { ElMessage.error('更新失败') }
}

async function loadImage() {
  if (results.value.length === 0) return
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.src = `/api/images/serve/${currentImgIdx.value}?t=${Date.now()}`
  img.onload = () => {
    imageObj.value = img
    const containerWidth = canvasContainer.value?.clientWidth || 900
    const maxHeight = 700
    let w = containerWidth - 320
    let h = (img.naturalHeight / img.naturalWidth) * w
    if (h > maxHeight) { h = maxHeight; w = (img.naturalWidth / img.naturalHeight) * h }
    canvasWidth.value = Math.floor(w)
    canvasHeight.value = Math.floor(h)
    scaleX.value = w / img.naturalWidth
    scaleY.value = h / img.naturalHeight
  }
}

async function loadResults() {
  try {
    const res = await captionApi.results()
    results.value = res.data.results || []
    if (results.value.length > 0) { await nextTick(); await loadImage() }
  } catch { results.value = [] }
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

watch(currentImgIdx, async () => { await nextTick(); await loadImage() })
onMounted(loadResults)
</script>

<style scoped>
.caption-page { max-width: 1500px; margin: 0 auto; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.page-header h2 { margin: 0; }
.header-actions { display: flex; gap: 12px; align-items: center; }
.empty-state { margin-top: 60px; }

.caption-workspace { display: flex; gap: 16px; }
.canvas-container { flex: 1; overflow: auto; border: 1px solid #e0e0e0; border-radius: 4px; background: #f5f5f5; }

.caption-panel { width: 320px; flex-shrink: 0; border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; max-height: 700px; overflow-y: auto; }
.caption-panel h3 { margin: 0 0 10px 0; font-size: 14px; }
</style>