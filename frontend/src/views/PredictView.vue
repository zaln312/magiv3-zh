<template>
  <div class="predict-page">
    <div class="page-header">
      <h2>Predict 预测结果</h2>
      <div class="header-actions">
        <el-select v-model="currentImgIdx" placeholder="选择图片" style="width: 200px">
          <el-option v-for="(_, idx) in results" :key="idx" :label="`图片 ${idx + 1}`" :value="idx" />
        </el-select>
        <el-button @click="addCharacterMode = !addCharacterMode" :type="addCharacterMode ? 'warning' : 'default'">
          {{ addCharacterMode ? '取消添加' : '添加人物框' }}
        </el-button>
        <el-button type="primary" @click="runCaption" :loading="captionLoading">
          执行 Caption
        </el-button>
      </div>
    </div>

    <div v-if="results.length === 0" class="empty-state">
      <el-empty description="暂无 Predict 结果，请先执行 Predict" />
    </div>

    <div v-else class="predict-workspace">
      <div class="canvas-container" ref="canvasContainer">
        <v-stage ref="stageRef" :config="stageConfig"
          @mousedown="handleStageMouseDown"
          @mousemove="handleStageMouseMove"
          @mouseup="handleStageMouseUp"
        >
          <v-layer>
            <v-image :config="imageConfig" />

            <v-rect v-for="(box, idx) in currentTextBoxes" :key="'tbox-' + idx"
              :config="getTextBoxConfig(idx, box)" />

            <v-line v-for="(assoc, idx) in currentAssociations" :key="'cline-' + idx"
              :config="getCharLineConfig(assoc)" />

            <v-rect v-for="(box, idx) in currentCharacters" :key="'cbox-' + idx"
              :config="getCharBoxConfig(idx, box)" />

            <v-text v-for="(box, idx) in currentCharacters" :key="'clabel-' + idx"
              :config="getCharLabelConfig(idx, box)" />
          </v-layer>
        </v-stage>
      </div>

      <div class="side-panels">
        <div class="char-list-panel">
          <h3>角色列表</h3>
          <div v-for="(char, idx) in currentCharacters" :key="'char-' + idx"
            class="char-item"
            :class="{ active: selectedCharIdx === idx }"
            :style="{ borderLeftColor: getColor(currentGlobalIds[idx]) }"
            @click="selectCharacter(idx)"
          >
            <div class="char-header">
              <span class="char-id" :style="{ background: getColor(currentGlobalIds[idx]) }">
                {{ currentGlobalIds[idx] }}
              </span>
              <el-input
                v-model="charNameMap[currentGlobalIds[idx]]"
                size="small"
                placeholder="角色名"
                @change="updateCharName(currentGlobalIds[idx], charNameMap[currentGlobalIds[idx]])"
                @click.stop
              />
              <el-button size="small" circle :icon="Delete" @click.stop="deleteCharacter(idx)" />
            </div>

            <div v-if="selectedCharIdx === idx" class="char-associations">
              <div class="assoc-title">关联文本框：</div>
              <el-checkbox-group v-model="selectedAssociations[idx]" @change="onAssocChange(idx)">
                <el-checkbox v-for="tIdx in currentTextBoxes.keys()" :key="tIdx" :value="tIdx" :label="tIdx">
                  [{{ tIdx }}] {{ (currentOcrTexts[tIdx] || '').slice(0, 10) }}
                </el-checkbox>
              </el-checkbox-group>
            </div>
          </div>
        </div>

        <div class="global-char-panel">
          <h3>全局角色库</h3>
          <div v-for="entry in globalCharLibrary" :key="entry.global_id" class="global-char-item">
            <span class="global-id-tag" :style="{ background: getColor(entry.global_id) }">
              {{ entry.global_id }}
            </span>
            <el-input
              v-model="charNameMap[entry.global_id]"
              size="small"
              placeholder="角色名"
              @change="updateCharName(entry.global_id, charNameMap[entry.global_id])"
            />
          </div>
          <el-empty v-if="!globalCharLibrary.length" description="暂无全局角色" :image-size="40" />
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
import { predictApi, captionApi, characterApi } from '../api/endpoints'

const router = useRouter()

const COLORS = ['#E00000', '#00CE00', '#0000FF', '#DBDB06', '#DD00DD', '#00E0E0',
  '#FFA500', '#800080', '#008000', '#000080', '#A52A2A', '#D8A4AD']

function getColor(id: number) {
  return COLORS[id % COLORS.length]
}

const results = ref<any[]>([])
const currentImgIdx = ref(0)
const selectedCharIdx = ref<number | null>(null)
const addCharacterMode = ref(false)
const captionLoading = ref(false)
const charNameMap = ref<Record<number, string>>({})
const globalCharLibrary = ref<any[]>([])

const stageRef = ref<any>(null)
const canvasContainer = ref<HTMLElement | null>(null)
const imageObj = ref<HTMLImageElement | null>(null)
const canvasWidth = ref(900)
const canvasHeight = ref(600)
const scaleX = ref(1)
const scaleY = ref(1)

const selectedAssociations = ref<Record<number, number[]>>({})

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

function getTextBoxConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0]), y = toCanvasY(box[1])
  return {
    x, y,
    width: toCanvasX(box[2]) - x,
    height: toCanvasY(box[3]) - y,
    stroke: '#00ff00', strokeWidth: 1,
    fill: 'rgba(0,255,0,0.03)',
    name: 'tbox-' + idx,
  }
}

function getCharBoxConfig(idx: number, box: number[]) {
  const x = toCanvasX(box[0]), y = toCanvasY(box[1])
  const gid = currentGlobalIds.value[idx] ?? idx
  return {
    x, y,
    width: toCanvasX(box[2]) - x,
    height: toCanvasY(box[3]) - y,
    stroke: getColor(gid),
    strokeWidth: selectedCharIdx.value === idx ? 3 : 2,
    fill: 'rgba(0,0,0,0)',
    name: 'cbox-' + idx,
  }
}

function getCharLabelConfig(idx: number, box: number[]) {
  const gid = currentGlobalIds.value[idx] ?? idx
  return {
    x: toCanvasX(box[0]),
    y: Math.max(0, toCanvasY(box[1]) - 18),
    text: `${gid}`,
    fontSize: 14,
    fill: '#fff',
    name: 'clabel-' + idx,
  }
}

function getCharLineConfig(assoc: number[]) {
  const tIdx = assoc[0], cIdx = assoc[1]
  const tBox = currentTextBoxes.value[tIdx]
  const cBox = currentCharacters.value[cIdx]
  if (!tBox || !cBox) return { points: [0, 0, 0, 0], stroke: '#999', strokeWidth: 1, dash: [4, 4] }
  const tx = toCanvasX((tBox[0] + tBox[2]) / 2)
  const ty = toCanvasY((tBox[1] + tBox[3]) / 2)
  const cx = toCanvasX((cBox[0] + cBox[2]) / 2)
  const cy = toCanvasY((cBox[1] + cBox[3]) / 2)
  const gid = currentGlobalIds.value[cIdx] ?? cIdx
  return {
    points: [tx, ty, cx, cy],
    stroke: getColor(gid),
    strokeWidth: 1.5,
    dash: [6, 4],
    name: 'cline-' + tIdx + '-' + cIdx,
  }
}

function selectCharacter(idx: number) {
  if (selectedCharIdx.value === idx) {
    selectedCharIdx.value = null
  } else {
    selectedCharIdx.value = idx
  }
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
    predictApi.addCharacter(currentImgIdx.value, box).then(() => {
      results.value[currentImgIdx.value].characters.push(box)
      const gids = results.value[currentImgIdx.value].global_character_ids
      gids.push(gids.length > 0 ? Math.max(...gids) + 1 : 0)
      ElMessage.success('已添加人物框')
    })
    addCharacterMode.value = false
    return
  }
  if (name && name.startsWith('cbox-')) {
    selectCharacter(parseInt(name.split('-')[1]))
  }
}

function handleStageMouseMove(_e: any) {}
function handleStageMouseUp(_e: any) {}

async function deleteCharacter(idx: number) {
  try {
    await predictApi.deleteCharacter(currentImgIdx.value, idx)
    results.value[currentImgIdx.value].characters.splice(idx, 1)
    results.value[currentImgIdx.value].global_character_ids.splice(idx, 1)
    if (selectedCharIdx.value === idx) selectedCharIdx.value = null
    ElMessage.success('已删除人物框')
  } catch { ElMessage.error('删除失败') }
}

async function updateCharName(globalId: number, name: string) {
  try {
    await characterApi.updateName(globalId, name)
  } catch { ElMessage.error('更新角色名失败') }
}

function onAssocChange(charIdx: number) {
  const textIndices = selectedAssociations.value[charIdx] || []
  const currentAssocs: [number, number][] = results.value[currentImgIdx.value].text_character_associations || []

  const newAssocs = currentAssocs.filter(([t, c]: [number, number]) => c !== charIdx)
  for (const tIdx of textIndices) {
    newAssocs.push([tIdx, charIdx])
  }
  results.value[currentImgIdx.value].text_character_associations = newAssocs

  predictApi.updateAssociation(currentImgIdx.value, -1, -1).catch(() => {})
  for (const [t, c] of newAssocs) {
    predictApi.updateAssociation(currentImgIdx.value, t, c).catch(() => {})
  }
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

async function runCaption() {
  captionLoading.value = true
  try {
    await captionApi.run()
    ElMessage.success('Caption 完成')
    router.push('/caption')
  } catch (e: any) {
    ElMessage.error('Caption 失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    captionLoading.value = false
  }
}

watch(currentImgIdx, async () => {
  selectedCharIdx.value = null
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

.predict-workspace { display: flex; gap: 16px; }
.canvas-container { flex: 1; overflow: auto; border: 1px solid #e0e0e0; border-radius: 4px; background: #f5f5f5; }

.side-panels { width: 300px; flex-shrink: 0; display: flex; flex-direction: column; gap: 12px; }

.char-list-panel, .global-char-panel {
  border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px;
  max-height: 400px; overflow-y: auto;
}
.char-list-panel h3, .global-char-panel h3 { margin: 0 0 10px 0; font-size: 14px; }

.char-item {
  border-left: 3px solid #ccc; padding: 6px 8px; margin-bottom: 6px;
  border-radius: 0 4px 4px 0; cursor: pointer; transition: background 0.2s;
}
.char-item:hover { background: #f5f5f5; }
.char-item.active { background: #e6f0ff; }

.char-header { display: flex; align-items: center; gap: 6px; }
.char-id {
  width: 28px; height: 28px; border-radius: 50%; color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: bold; flex-shrink: 0;
}
.char-header .el-input { flex: 1; }

.char-associations { margin-top: 8px; padding-top: 8px; border-top: 1px dashed #ddd; }
.assoc-title { font-size: 12px; color: #666; margin-bottom: 4px; }
.char-associations .el-checkbox { display: block; margin-bottom: 2px; font-size: 12px; }

.global-char-item { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.global-id-tag {
  width: 28px; height: 28px; border-radius: 50%; color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: bold; flex-shrink: 0;
}
.global-char-item .el-input { flex: 1; }
</style>