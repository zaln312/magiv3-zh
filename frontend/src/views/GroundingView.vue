<template>
  <div class="grounding-page">
    <div class="sub-nav">
      <div
        class="sub-nav-item"
        :class="{ active: currentStep === 0 }"
        @click="currentStep = 0"
      >
        风格化设置
      </div>
      <div
        class="sub-nav-item"
        :class="{ active: currentStep === 1 }"
        @click="currentStep = 1"
      >
        Grounding 指代结果
      </div>
    </div>

    <div v-show="currentStep === 0" class="style-step">
      <div class="style-card">
        <h3>风格化描述设置</h3>
        <p class="style-desc">
          输入风格化文本，用于指导多模态语言模型在描述每一张 panel 画面时使用自定义风格化的语言输出。默认为空时将使用标准描述风格。
        </p>

        <div class="style-form">
          <div class="style-form-row">
            <label>预设：</label>
            <el-select
              v-model="selectedPreset"
              placeholder="选择预设风格"
              style="width: 320px"
              @change="onPresetChange"
              clearable
            >
              <el-option
                v-for="preset in stylePresets"
                :key="preset.label"
                :label="preset.label"
                :value="preset.label"
              />
            </el-select>
          </div>

          <div class="style-form-row">
            <label>Prompt：</label>
            <el-input
              v-model="stylePrompt"
              type="textarea"
              :rows="6"
              placeholder="请输入自定义的风格化描述指令，例如：请用古龙武侠小说的风格描述画面..."
              resize="vertical"
            />
          </div>
        </div>

        <div class="style-actions">
          <el-button type="primary" @click="goNext" :loading="groundingLoading">
            下一步
          </el-button>
        </div>
      </div>
    </div>

    <div v-show="currentStep === 1">
      <div class="page-header">
        <h2>Grounding 指代</h2>
        <div class="header-actions">
          <el-select v-model="currentImgIdx" placeholder="选择图片" style="width: 200px">
            <el-option
              v-for="(_, idx) in results"
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
          <el-select v-model="currentPanelIdx" placeholder="选择分镜" style="width: 160px">
            <el-option
              v-for="(_, idx) in currentPanels"
              :key="idx"
              :label="`分镜 ${idx + 1}`"
              :value="idx"
            >
              <div
                style="margin: -8px -20px; padding: 8px 20px;"
                @mouseenter="onPanelOptionEnter(idx, $event)"
                @mouseleave="onPreviewLeave"
              >
                分镜 {{ idx + 1 }}
              </div>
            </el-option>
          </el-select>
          <el-button type="primary" @click="runProse" :loading="proseLoading">
            生成 Prose
          </el-button>
        </div>
      </div>

      <div v-if="results.length === 0" class="empty-state">
        <el-empty description="暂无 Grounding 结果，请先执行 Grounding" />
      </div>

      <div v-else class="grounding-workspace" ref="workspaceRef">
        <div class="canvas-container" ref="canvasContainer">
          <v-stage ref="stageRef" :config="stageConfig">
            <v-layer>
              <v-image :config="imageConfig" />

              <v-rect v-for="(box, idx) in currentPanelCharacters" :key="'cbox-' + idx"
                :config="getCharBoxConfig(idx, box)" />

              <v-rect v-for="(box, idx) in currentPanelCharacters" :key="'clabel-bg-' + idx"
                :config="getCharLabelBgConfig(idx, box)" />

              <v-text v-for="(box, idx) in currentPanelCharacters" :key="'clabel-' + idx"
                :config="getCharLabelConfig(idx, box)" />
            </v-layer>
          </v-stage>
        </div>

        <div class="caption-panel">
          <h3>Grounded Caption</h3>
          <div
            ref="editableRef"
            class="caption-editable"
            contenteditable="true"
            @input="onEditableInput"
            @blur="onEditableBlur"
            @click="onEditableClick"
            @mouseup="onEditableMouseUp"
          ></div>
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
      <div v-if="previewType === 'image'" style="width: 220px; overflow: hidden; border-radius: 4px; line-height: 0;">
        <img
          :key="previewIdx"
          :src="`/api/images/serve/${previewIdx}?t=${Date.now()}`"
          style="width: 100%; display: block;"
        />
      </div>
      <div v-else style="position: relative; width: 220px; overflow: hidden; border-radius: 4px; line-height: 0;">
        <img
          :src="`/api/images/serve/${currentImgIdx}?t=${Date.now()}`"
          style="width: 100%; display: block;"
        />
        <div
          v-if="panelOverlayStyles[previewIdx]"
          style="position: absolute; background: transparent; pointer-events: none; box-shadow: 0 0 0 9999px rgba(0,0,0,0.55);"
          :style="panelOverlayStyles[previewIdx]"
        />
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { groundingApi, proseApi, characterApi } from '../api/endpoints'

const router = useRouter()

const COLORS = ['#FF6B6B', '#3E7BFF', '#FF9E4D', '#C77DFF', '#FF4E9F', '#5E5CFF', '#FFBD7A', '#D96EFF', '#FF4D7E', '#4A8CFF', '#E07BFF', '#FF6B9D']

function getColor(id: number) {
  return COLORS[id % COLORS.length]
}

const currentStep = ref(0)
const stylePrompt = ref('')
const selectedPreset = ref('')
const groundingLoading = ref(false)

const stylePresets = [
  { label: '专业电影分镜描述', prompt: '你是一位电影摄影师和分镜师。请用专业术语描述这张图片，必须包含：景别（远景/全景/中景/近景/特写/大特写）、镜头角度与机位（俯拍/仰拍/平视/鸟瞰/过肩/低角度等）、镜头运动建议（推/拉/摇/移/跟/升/降）、构图与透视（三分法、引导线、深度感）、光线与色彩（光源方向、质感、色调）。用一段连贯的文字写成分镜说明，就像写在分镜表上的描述。' },
  { label: '朴素客观描述', prompt: '用完全客观、朴素的语言描述这张图片。只陈述可见的人、物、场景、颜色、位置关系、动作、天气等事实信息。不要使用任何比喻、拟人、情感词、夸张手法，就像在给一份档案写图片说明。' },
  { label: '童话风格', prompt: '用经典童话的口吻来描述这张图片：语气温柔天真，可以使用拟人、排比、叠词，把万物都当成有生命有感情的存在。句子要简短，像在给孩子讲故事一样。' },
  { label: '武侠风格', prompt: '以金庸或古龙式的武侠文风描述这张图片。语言要凝练、有意境，多用短句、留白。可以描写人物的"内力""气机""身法"等虚拟感觉，环境要有江湖萧瑟或风起云涌之感。景物可视为"江湖一隅"，给场景注入侠客的气氛。' },
  { label: '科幻风格', prompt: '以硬科幻或赛博朋克小说作者的笔触描述这张图片。将图中一切解释为未来科技产物：建筑是"纳米聚合体"，灯光是"全息投影或数据流"，人物可能是"仿生人、AI或星际居民"。强调材质（合成材料、金属、冷光）、数据感、社会结构暗示。用冷静又带点疏离的语气写。' },
]

function onPresetChange(value: string) {
  if (!value) {
    stylePrompt.value = ''
    return
  }
  const preset = stylePresets.find(p => p.label === value)
  if (preset) {
    stylePrompt.value = preset.prompt
  }
}

const results = ref<any[]>([])
const currentImgIdx = ref(0)
const currentPanelIdx = ref(0)
const proseLoading = ref(false)
const panelCharacters = ref<any[][]>([])

const stageRef = ref<any>(null)
const canvasContainer = ref<HTMLElement | null>(null)
const workspaceRef = ref<HTMLElement | null>(null)
const editableRef = ref<HTMLElement | null>(null)
const imageObj = ref<HTMLImageElement | null>(null)
const canvasWidth = ref(600)
const canvasHeight = ref(400)
const isComposing = ref(false)
const selectedCharId = ref<number | null>(null)
let renderGuard = false

const previewVisible = ref(false)
const previewType = ref<'image' | 'panel'>('image')
const previewIdx = ref(0)
const previewStyle = ref({ top: '0px', left: '0px' })
let hideTimer: ReturnType<typeof setTimeout> | null = null

function onImgOptionEnter(idx: number, e: MouseEvent) {
  if (hideTimer) { clearTimeout(hideTimer); hideTimer = null }
  previewType.value = 'image'
  previewIdx.value = idx
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
  previewStyle.value = {
    top: rect.top + 'px',
    left: (rect.left - 232) + 'px',
  }
  previewVisible.value = true
}

function onPanelOptionEnter(idx: number, e: MouseEvent) {
  if (hideTimer) { clearTimeout(hideTimer); hideTimer = null }
  previewType.value = 'panel'
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

const currentPanels = computed(() => {
  if (currentImgIdx.value >= results.value.length) return []
  return results.value[currentImgIdx.value].panels || []
})

const currentPanel = computed(() => {
  const panels = currentPanels.value
  if (currentPanelIdx.value >= panels.length) return null
  return panels[currentPanelIdx.value]
})

const currentGroundedCaption = computed({
  get: () => {
    const caps = results.value[currentImgIdx.value]?.grounded_captions_per_panel
    if (!caps || currentPanelIdx.value >= caps.length) return ''
    return caps[currentPanelIdx.value] || ''
  },
  set: (val) => {
    const caps = results.value[currentImgIdx.value]?.grounded_captions_per_panel
    if (caps && currentPanelIdx.value < caps.length) {
      caps[currentPanelIdx.value] = val
    }
  },
})

const currentPanelCharacters = computed(() => {
  if (currentImgIdx.value >= panelCharacters.value.length) return []
  const panels = panelCharacters.value[currentImgIdx.value]
  if (!panels || currentPanelIdx.value >= panels.length) return []
  return panels[currentPanelIdx.value] || []
})

const panelOverlayStyles = computed(() => {
  const img = imageObj.value
  if (!img) return {}
  const imgW = img.naturalWidth
  const imgH = img.naturalHeight
  if (!imgW || !imgH) return {}
  const styles: Record<number, Record<string, string>> = {}
  currentPanels.value.forEach((panel: number[], idx: number) => {
    const px = (panel[0] / imgW) * 100
    const py = (panel[1] / imgH) * 100
    const pw = ((panel[2] - panel[0]) / imgW) * 100
    const ph = ((panel[3] - panel[1]) / imgH) * 100
    styles[idx] = {
      left: px + '%',
      top: py + '%',
      width: pw + '%',
      height: ph + '%',
    }
  })
  return styles
})

function buildColoredHtml(text: string): string {
  if (!text) return ''
  const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  return escaped.replace(/\[(\d+)\]/g, (_match: string, id: string) => {
    const color = getColor(parseInt(id))
    return `<span style="color:${color};font-weight:bold;">[${id}]</span>`
  })
}

function getPlainText(el: HTMLElement): string {
  return (el.textContent || '').replace(/\u00A0/g, ' ')
}

function applyColoring(el: HTMLElement) {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT)
  const textNodes: Text[] = []
  let node = walker.nextNode()
  while (node) {
    textNodes.push(node as Text)
    node = walker.nextNode()
  }

  for (const textNode of textNodes) {
    if (!textNode.parentElement) continue
    const text = textNode.textContent || ''
    const parent = textNode.parentElement

    const isInColoredSpan = parent !== el &&
      parent.tagName === 'SPAN' &&
      parent.style.color !== '' &&
      parent.style.fontWeight === 'bold'

    const hasIdPattern = /\[(\d+)\]/.test(text)

    if (isInColoredSpan && !hasIdPattern) {
      parent.replaceWith(document.createTextNode(text))
    } else if (!isInColoredSpan && hasIdPattern) {
      const fragment = document.createDocumentFragment()
      let lastIndex = 0
      const regex = /\[(\d+)\]/g
      let match
      while ((match = regex.exec(text)) !== null) {
        if (match.index > lastIndex) {
          fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.index)))
        }
        const span = document.createElement('span')
        span.style.color = getColor(parseInt(match[1]))
        span.style.fontWeight = 'bold'
        span.textContent = match[0]
        fragment.appendChild(span)
        lastIndex = regex.lastIndex
      }
      if (lastIndex < text.length) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex)))
      }
      textNode.replaceWith(fragment)
    }
  }
}

function onEditableInput(_e: Event) {
  if (isComposing.value || renderGuard) return
  const el = editableRef.value
  if (!el) return

  renderGuard = true
  applyColoring(el)
  currentGroundedCaption.value = getPlainText(el)

  const sel = window.getSelection()
  if (sel && sel.isCollapsed && sel.anchorNode) {
    let node: Node | null = sel.anchorNode
    let found = false
    while (node && node !== el) {
      if (node.nodeType === Node.ELEMENT_NODE) {
        const elem = node as HTMLElement
        if (elem.tagName === 'SPAN' && elem.style.color && elem.style.fontWeight === 'bold') {
          const match = elem.textContent?.match(/^\[(\d+)\]$/)
          if (match) {
            const id = parseInt(match[1])
            selectSpanInText(elem)
            selectedCharId.value = id
            found = true
            break
          }
        }
      }
      node = node.parentNode
    }
    if (!found) {
      selectedCharId.value = null
    }
  } else {
    selectedCharId.value = null
  }

  renderGuard = false
}

function onEditableBlur() {
  updateCaption()
}

function syncEditableHtml() {
  const el = editableRef.value
  if (el) {
    el.innerHTML = buildColoredHtml(currentGroundedCaption.value)
  }
}

const stageConfig = computed(() => ({
  width: canvasWidth.value,
  height: canvasHeight.value,
}))

const imageConfig = computed(() => {
  const panel = currentPanel.value
  if (!panel || !imageObj.value) {
    return { image: imageObj.value, width: canvasWidth.value, height: canvasHeight.value }
  }
  const cropX = panel[0]
  const cropY = panel[1]
  const cropW = panel[2] - panel[0]
  const cropH = panel[3] - panel[1]
  return {
    image: imageObj.value,
    cropX,
    cropY,
    cropWidth: cropW,
    cropHeight: cropH,
    width: canvasWidth.value,
    height: canvasHeight.value,
  }
})

function getCharBoxConfig(idx: number, char: any) {
  const box = char.local_box
  const gid = char.cluster_label
  const x = box[0] * (canvasWidth.value / (currentPanel.value[2] - currentPanel.value[0]))
  const y = box[1] * (canvasHeight.value / (currentPanel.value[3] - currentPanel.value[1]))
  const w = (box[2] - box[0]) * (canvasWidth.value / (currentPanel.value[2] - currentPanel.value[0]))
  const h = (box[3] - box[1]) * (canvasHeight.value / (currentPanel.value[3] - currentPanel.value[1]))
  const isSelected = selectedCharId.value === gid
  const shouldDim = selectedCharId.value !== null && !isSelected
  return {
    x, y, width: w, height: h,
    stroke: getColor(gid),
    strokeWidth: isSelected ? 3 : 2,
    fill: isSelected ? getColor(gid) + '20' : 'rgba(0,0,0,0)',
    opacity: shouldDim ? 0.4 : 1,
    name: 'cbox-' + idx,
  }
}

function getCharLabelBgConfig(idx: number, char: any) {
  const box = char.local_box
  const gid = char.cluster_label
  const x = box[0] * (canvasWidth.value / (currentPanel.value[2] - currentPanel.value[0]))
  const y = box[1] * (canvasHeight.value / (currentPanel.value[3] - currentPanel.value[1]))
  const labelY = Math.max(0, y - 18)
  const numDigits = String(gid).length
  const bgWidth = numDigits * 7 + 6
  const isSelected = selectedCharId.value === gid
  const shouldDim = selectedCharId.value !== null && !isSelected
  return {
    x: x - 2,
    y: labelY,
    width: bgWidth,
    height: 14,
    fill: getColor(gid),
    cornerRadius: 3,
    opacity: shouldDim ? 0.4 : 1,
    name: 'clabel-bg-' + idx,
  }
}

function getCharLabelConfig(idx: number, char: any) {
  const box = char.local_box
  const gid = char.cluster_label
  const x = box[0] * (canvasWidth.value / (currentPanel.value[2] - currentPanel.value[0]))
  const y = box[1] * (canvasHeight.value / (currentPanel.value[3] - currentPanel.value[1]))
  const labelY = Math.max(0, y - 18)
  const isSelected = selectedCharId.value === gid
  const shouldDim = selectedCharId.value !== null && !isSelected
  return {
    x: x + 1,
    y: labelY + 1,
    text: `${gid}`,
    fontSize: 12,
    fontStyle: 'bold',
    fill: '#fff',
    opacity: shouldDim ? 0.4 : 1,
    name: 'clabel-' + idx,
  }
}

function findCharSpanAtPoint(el: HTMLElement, clientX: number, clientY: number): HTMLElement | null {
  const range = document.caretRangeFromPoint(clientX, clientY)
  if (!range) return null
  let node: Node | null = range.startContainer
  while (node && node !== el) {
    if (node.nodeType === Node.ELEMENT_NODE) {
      const elem = node as HTMLElement
      if (elem.tagName === 'SPAN' && elem.style.color && elem.style.fontWeight === 'bold') {
        const match = elem.textContent?.match(/^\[(\d+)\]$/)
        if (match) return elem
      }
    }
    node = node.parentNode
  }
  return null
}

function selectSpanInText(span: HTMLElement) {
  const range = document.createRange()
  range.selectNodeContents(span)
  const sel = window.getSelection()
  if (sel) {
    sel.removeAllRanges()
    sel.addRange(range)
  }
}

function onEditableClick(e: MouseEvent) {
  const el = editableRef.value
  if (!el) return
  const span = findCharSpanAtPoint(el, e.clientX, e.clientY)
  if (span) {
    const match = span.textContent?.match(/^\[(\d+)\]$/)
    const charId = match ? parseInt(match[1]) : null
    if (charId !== null && selectedCharId.value === charId) return
    selectedCharId.value = charId
    selectSpanInText(span)
  } else {
    selectedCharId.value = null
  }
}

function onEditableMouseUp(_e: MouseEvent) {
  const el = editableRef.value
  if (!el) return
  const sel = window.getSelection()
  if (!sel || sel.isCollapsed) {
    selectedCharId.value = null
    return
  }
  const anchorNode = sel.anchorNode
  if (!anchorNode || !el.contains(anchorNode)) return
  let node: Node | null = anchorNode
  while (node && node !== el) {
    if (node.nodeType === Node.ELEMENT_NODE) {
      const elem = node as HTMLElement
      if (elem.tagName === 'SPAN' && elem.style.color && elem.style.fontWeight === 'bold') {
        const match = elem.textContent?.match(/^\[(\d+)\]$/)
        if (match) {
          const id = parseInt(match[1])
          if (selectedCharId.value !== id) {
            selectedCharId.value = id
          }
          return
        }
      }
    }
    node = node.parentNode
  }
  selectedCharId.value = null
}

async function updateCaption() {
  try {
    await groundingApi.updateCaption(currentImgIdx.value, currentPanelIdx.value, currentGroundedCaption.value)
  } catch { ElMessage.error('更新失败') }
}

async function loadImage() {
  if (results.value.length === 0) return
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.src = `/api/images/serve/${currentImgIdx.value}?t=${Date.now()}`
  img.onload = () => {
    imageObj.value = img
    const panel = currentPanel.value
    if (!panel) return
    const panelW = panel[2] - panel[0]
    const panelH = panel[3] - panel[1]
    const workspaceWidth = workspaceRef.value?.clientWidth || 1200
    const maxHeight = 700
    const sidePanelWidth = 340
    const gap = 16
    let w = workspaceWidth - sidePanelWidth - gap
    let h = (panelH / panelW) * w
    if (h > maxHeight) {
      h = maxHeight
      w = (panelW / panelH) * h
    }
    canvasWidth.value = Math.floor(w)
    canvasHeight.value = Math.floor(h)
  }
}

async function loadResults() {
  try {
    const res = await groundingApi.results()
    results.value = res.data.results || []
    const hasGrounded = results.value.some((r: any) =>
      r.grounded_captions_per_panel && r.grounded_captions_per_panel.length > 0
    )
    if (results.value.length > 0 && hasGrounded) {
      await loadPanelCharacters()
      syncEditableHtml()
      await nextTick()
      await loadImage()
    }
  } catch { results.value = [] }
}

async function loadPanelCharacters() {
  try {
    const allPanelChars: any[][] = []
    for (let i = 0; i < results.value.length; i++) {
      const res = await characterApi.panelCharacters(i)
      allPanelChars.push(res.data.panel_characters || [])
    }
    panelCharacters.value = allPanelChars
  } catch {}
}

async function goNext() {
  groundingLoading.value = true
  try {
    await groundingApi.run(stylePrompt.value || undefined)
    ElMessage.success('Grounding 完成')
    currentStep.value = 1
    await loadResults()
  } catch (e: any) {
    ElMessage.error('Grounding 失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    groundingLoading.value = false
  }
}

async function runProse() {
  proseLoading.value = true
  try {
    await proseApi.run()
    ElMessage.success('Prose 生成完成')
    router.push('/prose')
  } catch (e: any) {
    ElMessage.error('生成失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    proseLoading.value = false
  }
}

watch(currentImgIdx, () => {
  currentPanelIdx.value = 0
})

watch([currentImgIdx, currentPanelIdx], async () => {
  selectedCharId.value = null
  syncEditableHtml()
  await nextTick()
  await loadImage()
})

onMounted(loadResults)
</script>

<style scoped>
.grounding-page { max-width: 1500px; margin: 0 auto; }

.sub-nav {
  display: flex;
  gap: 0;
  border-bottom: 2px solid #e0e0e0;
  margin-bottom: 20px;
}
.sub-nav-item {
  padding: 10px 24px;
  font-size: 14px;
  font-weight: 500;
  color: #666;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  transition: all 0.2s;
  user-select: none;
}
.sub-nav-item:hover {
  color: #409eff;
}
.sub-nav-item.active {
  color: #409eff;
  border-bottom-color: #409eff;
}

.style-step {
  display: flex;
  justify-content: center;
  padding-top: 20px;
}
.style-card {
  width: 640px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 28px 32px;
  background: #fff;
}
.style-card h3 {
  margin: 0 0 8px 0;
  font-size: 17px;
  color: #303133;
}
.style-desc {
  margin: 0 0 24px 0;
  font-size: 13px;
  color: #909399;
  line-height: 1.6;
}
.style-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 24px;
}
.style-form-row {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.style-form-row label {
  flex-shrink: 0;
  width: 100px;
  text-align: right;
  line-height: 32px;
  font-size: 14px;
  color: #606266;
}
.style-actions {
  display: flex;
  justify-content: flex-end;
}

.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.page-header h2 { margin: 0; }
.header-actions { display: flex; gap: 12px; align-items: center; }
.empty-state { margin-top: 60px; }

.grounding-workspace { display: flex; gap: 16px; align-items: flex-start; }

.canvas-container {
  flex-shrink: 0;
  overflow: hidden;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background: #f5f5f5;
}

.caption-panel {
  width: 340px;
  flex-shrink: 0;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 12px;
  max-height: 700px;
  overflow-y: auto;
}
.caption-panel h3 {
  margin: 0 0 10px 0;
  font-size: 15px;
  position: sticky;
  top: 0;
  background: #fff;
  padding-bottom: 8px;
  z-index: 1;
}

.caption-editable {
  padding: 10px 12px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 14px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  min-height: 300px;
  max-height: 620px;
  overflow-y: auto;
  outline: none;
  background: #fff;
  transition: border-color 0.2s;
}
.caption-editable:focus {
  border-color: #409eff;
  box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.15);
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
</style>