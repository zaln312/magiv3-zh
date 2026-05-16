<template>
  <div class="grounding-page">
    <div class="page-header">
      <h2>Grounding 结果</h2>
      <div class="header-actions">
        <el-select v-model="currentImgIdx" placeholder="选择图片" style="width: 200px">
          <el-option v-for="(_, idx) in results" :key="idx" :label="`图片 ${idx + 1}`" :value="idx" />
        </el-select>
        <el-button type="primary" @click="runProsePrompt" :loading="prosePromptLoading">
          构建 Prose Prompt
        </el-button>
      </div>
    </div>

    <div v-if="results.length === 0" class="empty-state">
      <el-empty description="暂无 Grounding 结果，请先执行 Grounding" />
    </div>

    <div v-else class="grounding-workspace">
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

      <div class="grounding-panel">
        <h3>Grounded Caption</h3>
        <div v-if="currentGroundedCaption" class="grounded-text">
          <el-input
            v-model="currentGroundedCaption"
            type="textarea"
            :rows="14"
            resize="none"
          />
        </div>
        <el-empty v-else description="暂无 Grounded Caption" :image-size="40" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { groundingApi, proseApi } from '../api/endpoints'

const router = useRouter()

const COLORS = ['#E00000', '#00CE00', '#0000FF', '#DBDB06', '#DD00DD', '#00E0E0',
  '#FFA500', '#800080', '#008000', '#000080', '#A52A2A', '#D8A4AD']

function getColor(id: number) { return COLORS[id % COLORS.length] }

const results = ref<any[]>([])
const currentImgIdx = ref(0)
const prosePromptLoading = ref(false)

const canvasContainer = ref<HTMLElement | null>(null)
const imageObj = ref<HTMLImageElement | null>(null)
const canvasWidth = ref(900)
const canvasHeight = ref(600)
const scaleX = ref(1)
const scaleY = ref(1)

const currentResult = computed(() => results.value[currentImgIdx.value] || {})
const currentGroundedCaption = computed({
  get: () => currentResult.value.grounded_caption || '',
  set: (val) => { if (results.value[currentImgIdx.value]) results.value[currentImgIdx.value].grounded_caption = val },
})
const currentTextBoxes = computed(() => currentResult.value.texts || [])
const currentCharacters = computed(() => currentResult.value.characters || [])
const currentGlobalIds = computed(() => currentResult.value.global_character_ids || [])
const currentAssociations = computed(() => currentResult.value.text_character_associations || [])

const stageConfig = computed(() => ({ width: canvasWidth.value, height: canvasHeight.value }))
const imageConfig = computed(() => ({ image: imageObj.value, width: canvasWidth.value, height: canvasHeight.value }))

function toCanvasX(x: number) { return x * scaleX.value }
function toCanvasY(y: number) { return y * scaleY.value }

function getTextBoxConfig(idx: number | string, box: any[]) {
  const x = toCanvasX(Number(box[0])), y = toCanvasY(Number(box[1]))
  return { x, y, width: toCanvasX(Number(box[2])) - x, height: toCanvasY(Number(box[3])) - y, stroke: '#00ff00', strokeWidth: 1, fill: 'rgba(0,255,0,0.03)', name: 'tbox-' + idx }
}

function getCharBoxConfig(idx: number | string, box: any[]) {
  const x = toCanvasX(Number(box[0])), y = toCanvasY(Number(box[1]))
  const gid = currentGlobalIds.value[Number(idx)] ?? Number(idx)
  return { x, y, width: toCanvasX(Number(box[2])) - x, height: toCanvasY(Number(box[3])) - y, stroke: getColor(gid), strokeWidth: 2, fill: 'rgba(0,0,0,0)', name: 'cbox-' + idx }
}

function getCharLabelConfig(idx: number | string, box: any[]) {
  const gid = currentGlobalIds.value[Number(idx)] ?? Number(idx)
  return { x: toCanvasX(Number(box[0])), y: Math.max(0, toCanvasY(Number(box[1])) - 18), text: `${gid}`, fontSize: 14, fill: '#fff', name: 'clabel-' + idx }
}

function getCharLineConfig(assoc: number[]) {
  const tIdx = assoc[0], cIdx = assoc[1]
  const tBox = currentTextBoxes.value[tIdx]
  const cBox = currentCharacters.value[cIdx]
  if (!tBox || !cBox) return { points: [0, 0, 0, 0], stroke: '#999', strokeWidth: 1, dash: [4, 4] }
  const gid = currentGlobalIds.value[cIdx] ?? cIdx
  return { points: [toCanvasX((tBox[0] + tBox[2]) / 2), toCanvasY((tBox[1] + tBox[3]) / 2), toCanvasX((cBox[0] + cBox[2]) / 2), toCanvasY((cBox[1] + cBox[3]) / 2)], stroke: getColor(gid), strokeWidth: 1.5, dash: [6, 4], name: 'cline-' + tIdx + '-' + cIdx }
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
    const res = await groundingApi.results()
    console.log('[DEBUG] GroundingView loadResults 原始响应:', res.data)
    console.log('[DEBUG] res.data.results:', res.data.results)
    console.log('[DEBUG] res.data.count:', res.data.count)
    results.value = res.data.results || []
    console.log('[DEBUG] results.value 长度:', results.value.length)
    if (results.value.length > 0) {
      console.log('[DEBUG] results[0] keys:', Object.keys(results.value[0]))
      console.log('[DEBUG] results[0].grounded_caption:', results.value[0].grounded_caption?.substring(0, 80))
    }
    if (results.value.length > 0) { await nextTick(); await loadImage() }
  } catch (e) {
    console.error('[DEBUG] GroundingView loadResults 异常:', e)
    results.value = []
  }
}

async function runProsePrompt() {
  prosePromptLoading.value = true
  try {
    await proseApi.buildScripts()
    await proseApi.buildPrompt()
    ElMessage.success('Prose Prompt 构建完成')
    router.push('/prose-prompt')
  } catch (e: any) {
    ElMessage.error('构建失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    prosePromptLoading.value = false
  }
}

watch(currentImgIdx, async () => { await nextTick(); await loadImage() })
onMounted(loadResults)
</script>

<style scoped>
.grounding-page { max-width: 1500px; margin: 0 auto; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.page-header h2 { margin: 0; }
.header-actions { display: flex; gap: 12px; align-items: center; }
.empty-state { margin-top: 60px; }

.grounding-workspace { display: flex; gap: 16px; }
.canvas-container { flex: 1; overflow: auto; border: 1px solid #e0e0e0; border-radius: 4px; background: #f5f5f5; }

.grounding-panel { width: 320px; flex-shrink: 0; border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; max-height: 700px; overflow-y: auto; }
.grounding-panel h3 { margin: 0 0 10px 0; font-size: 14px; }
</style>