<template>
  <div class="video-generation-page">
    <div class="sub-nav">
      <div
        class="sub-nav-item"
        :class="{ active: currentStep === 0 }"
        @click="currentStep = 0"
      >
        Prose 与参考图
      </div>
      <div
        class="sub-nav-item"
        :class="{ active: currentStep === 1 }"
        @click="currentStep = 1"
      >
        视频生成
      </div>
    </div>

    <div v-show="currentStep === 0" class="prepare-step">
      <div v-if="!prose && characters.length === 0" class="empty-state">
        <el-empty description="暂无数据，请先完成 Prose 叙述和人物参考图" />
      </div>

      <div v-else class="prepare-layout">
        <div class="prepare-left">
          <div class="section-header">
            <h3>Prose 叙述</h3>
            <el-button size="small" type="primary" @click="copyProse">
              一键复制
            </el-button>
          </div>
          <div class="prose-text-area">
            <el-input
              v-model="prose"
              type="textarea"
              :rows="18"
              resize="vertical"
              readonly
            />
          </div>
        </div>

        <div class="prepare-right">
          <div class="section-header">
            <h3>角色参考图</h3>
            <el-button size="small" type="primary" @click="exportAllImages">
              一键打包导出
            </el-button>
          </div>
          <div class="char-ref-list">
            <div
              v-for="char in characters"
              :key="char.global_id"
              class="char-ref-row"
            >
              <div class="char-ref-name">
                <span class="char-gid-tag">ID: {{ char.global_id }}</span>
                <span class="char-name-text">{{ char.name }}</span>
              </div>
              <div class="char-ref-images">
                <div
                  v-for="(ref, idx) in char.ref_images"
                  :key="idx"
                  class="char-ref-img-item"
                >
                  <img
                    v-if="ref.image_base64"
                    :src="'data:image/png;base64,' + ref.image_base64"
                    class="char-ref-thumb"
                  />
                  <div v-if="ref.view" class="char-view-tag">{{ viewLabel(ref.view) }}</div>
                </div>
                <div
                  v-for="(img, idx) in char.design_images"
                  :key="'d' + idx"
                  class="char-ref-img-item"
                >
                  <img
                    :src="'data:image/png;base64,' + img.image_base64"
                    class="char-ref-thumb"
                  />
                  <div class="char-view-tag">设定集</div>
                </div>
                <div v-if="char.ref_images.length === 0 && char.design_images.length === 0" class="char-no-images">
                  暂无参考图
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-if="prose || characters.length > 0" class="prepare-next-action">
        <el-button type="primary" size="large" @click="currentStep = 1">
          下一步
        </el-button>
      </div>
    </div>

    <div v-show="currentStep === 1" class="generate-step">
      <div class="generate-card">
        <h3>视频生成</h3>
        <p class="generate-desc">
          基于 Prose 叙述和角色参考图，提交视频生成任务并自动轮询状态。提交后可返回首页，状态将同步显示在项目列表中。
        </p>

        <div v-if="!taskId && !taskState && !videoResult" class="generate-controls">
          <el-button
            type="primary"
            size="large"
            :loading="submitting"
            @click="submitTask"
          >
            提交视频生成任务
          </el-button>
        </div>

        <div v-if="taskState && taskState !== 'success' && taskState !== 'failed'" class="task-status-area">
          <div class="task-info">
            <div class="task-id-label">任务 ID：</div>
            <div class="task-id-value">{{ taskId }}</div>
          </div>
          <div class="task-state">
            <el-tag :type="stateTagType(taskState)" size="large">
              {{ stateLabel(taskState) }}
            </el-tag>
            <el-icon v-if="taskState !== 'failed'" class="is-loading poll-spinner"><Loading /></el-icon>
          </div>
          <div class="task-actions">
            <el-button @click="goHome">返回首页（后台继续轮询）</el-button>
            <el-button type="danger" plain @click="cancelTask" :loading="cancelling">取消任务</el-button>
          </div>
        </div>

        <div v-if="taskState === 'failed'" class="task-failed">
          <el-alert title="视频生成失败" type="error" :closable="false" show-icon />
          <div class="task-info" style="margin-top: 12px;">
            <div class="task-id-label">任务 ID：</div>
            <div class="task-id-value">{{ taskId }}</div>
          </div>
          <div class="task-actions" style="margin-top: 12px;">
            <el-button type="primary" @click="retryTask">重新提交</el-button>
          </div>
        </div>

        <div v-if="videoResult" class="video-result">
          <div class="video-player-wrapper">
            <video
              v-if="videoSrc"
              ref="videoPlayer"
              controls
              class="video-player"
              :src="videoSrc"
            >
              您的浏览器不支持 video 标签。
            </video>
          </div>
          <div v-if="creations.length > 0 && !videoSrc" class="creations-list">
            <div
              v-for="(creation, idx) in creations"
              :key="idx"
              class="creation-item"
            >
              <a v-if="creation.url" :href="creation.url" target="_blank">
                视频 {{ idx + 1 }}: {{ creation.url }}
              </a>
              <img
                v-if="creation.cover_url"
                :src="creation.cover_url"
                class="creation-cover"
              />
            </div>
          </div>
          <div class="video-actions">
            <el-button size="small" @click="downloadVideo" v-if="videoSrc">下载视频</el-button>
          </div>
        </div>

        <div v-if="generateError" class="generate-error">
          <el-alert :title="generateError" type="error" :closable="false" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Loading } from '@element-plus/icons-vue'
import { videoApi } from '../api/endpoints'

const router = useRouter()

const currentStep = ref(0)
const prose = ref('')
const submitting = ref(false)
const cancelling = ref(false)
const generateError = ref('')
const videoResult = ref<{ filename?: string; video_base64?: string; creations?: any[]; prose?: string } | null>(null)
const videoPlayer = ref<HTMLVideoElement | null>(null)
const taskId = ref('')
const taskState = ref('')
const creations = ref<{ url: string; cover_url: string }[]>([])
let pollTimer: ReturnType<typeof setInterval> | null = null

interface RefImg {
  image_base64: string
  view: string
}

interface DesignImg {
  filename: string
  image_base64: string
}

interface CharacterEntry {
  global_id: number
  name: string
  ref_images: RefImg[]
  design_images: DesignImg[]
}

const characters = ref<CharacterEntry[]>([])

const STATE_LABELS: Record<string, string> = {
  created: '创建成功',
  queueing: '排队中',
  processing: '处理中',
  success: '成功',
  failed: '失败',
}

const STATE_TAGS: Record<string, string> = {
  created: 'info',
  queueing: 'warning',
  processing: '',
  success: 'success',
  failed: 'danger',
}

function stateLabel(state: string): string {
  return STATE_LABELS[state] || state || '未知'
}

function stateTagType(state: string): '' | 'success' | 'warning' | 'info' | 'danger' {
  return (STATE_TAGS[state] || 'info') as '' | 'success' | 'warning' | 'info' | 'danger'
}

const VIEW_LABELS: Record<string, string> = {
  front: '正面',
  back: '反面',
  side: '侧面',
}

function viewLabel(v: string) {
  return VIEW_LABELS[v] || v
}

const videoSrc = computed(() => {
  if (!videoResult.value) return ''
  const b64 = videoResult.value.video_base64
  if (!b64) return ''
  return `data:video/mp4;base64,${b64}`
})

function startPolling() {
  stopPolling()
  pollTimer = setInterval(async () => {
    try {
      const res = await videoApi.status()
      if (res.data.success) {
        taskState.value = res.data.state || ''
        creations.value = res.data.creations || []
        if (taskState.value === 'success' || taskState.value === 'failed') {
          stopPolling()
          if (taskState.value === 'success') {
            await loadResult()
          }
        }
      }
    } catch {
    }
  }, 3000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function loadPrepareData() {
  try {
    const res = await videoApi.prepare()
    if (res.data.success) {
      prose.value = res.data.prose || ''
      characters.value = res.data.characters || []
    }
  } catch {
  }
}

async function loadResult() {
  try {
    const res = await videoApi.result()
    if (res.data?.video_result) {
      videoResult.value = res.data.video_result
    }
    if (res.data?.state) {
      taskState.value = res.data.state
    }
    if (res.data?.creations) {
      creations.value = res.data.creations
    }
  } catch {
  }
}

async function copyProse() {
  try {
    await navigator.clipboard.writeText(prose.value)
    ElMessage.success('Prose 文本已复制到剪贴板')
  } catch {
    ElMessage.error('复制失败，请手动复制')
  }
}

async function exportAllImages() {
  const JSZip = (await import('jszip')).default
  const zip = new JSZip()

  let imgCount = 0
  for (const char of characters.value) {
    const charName = char.name || `角色${char.global_id}`
    const folder = zip.folder(charName)

    for (let i = 0; i < char.ref_images.length; i++) {
      const ref = char.ref_images[i]
      if (ref.image_base64) {
        folder?.file(
          `ref_${viewLabel(ref.view)}_${i + 1}.png`,
          ref.image_base64,
          { base64: true }
        )
        imgCount++
      }
    }

    for (let i = 0; i < char.design_images.length; i++) {
      const img = char.design_images[i]
      if (img.image_base64) {
        folder?.file(
          `design_${img.filename || `img_${i + 1}`}.png`,
          img.image_base64,
          { base64: true }
        )
        imgCount++
      }
    }
  }

  if (imgCount === 0) {
    ElMessage.warning('没有可导出的参考图')
    return
  }

  const content = await zip.generateAsync({ type: 'blob' })
  const url = URL.createObjectURL(content)
  const a = document.createElement('a')
  a.href = url
  a.download = 'character_references.zip'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  ElMessage.success(`已导出 ${imgCount} 张参考图`)
}

async function submitTask() {
  submitting.value = true
  generateError.value = ''
  try {
    const res = await videoApi.submit()
    if (res.data.success) {
      taskId.value = res.data.task_id || ''
      taskState.value = res.data.state || 'created'
      creations.value = []
      ElMessage.success('视频生成任务已提交')
      startPolling()
    }
  } catch (e: any) {
    generateError.value = e.response?.data?.detail || '任务提交失败'
    ElMessage.error(generateError.value)
  } finally {
    submitting.value = false
  }
}

async function cancelTask() {
  cancelling.value = true
  try {
    await videoApi.cancel()
    stopPolling()
    taskId.value = ''
    taskState.value = ''
    creations.value = []
    ElMessage.success('任务已取消')
  } catch (e: any) {
    ElMessage.error('取消任务失败')
  } finally {
    cancelling.value = false
  }
}

function retryTask() {
  taskId.value = ''
  taskState.value = ''
  creations.value = []
  videoResult.value = null
  generateError.value = ''
}

function goHome() {
  router.push('/')
}

function downloadVideo() {
  if (!videoResult.value) return
  const b64 = videoResult.value.video_base64
  if (!b64) return

  const byteChars = atob(b64)
  const byteNums = new Array(byteChars.length)
  for (let i = 0; i < byteChars.length; i++) {
    byteNums[i] = byteChars.charCodeAt(i)
  }
  const byteArr = new Uint8Array(byteNums)
  const blob = new Blob([byteArr], { type: 'video/mp4' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = videoResult.value.filename || 'generated_video.mp4'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

onMounted(async () => {
  await loadPrepareData()
  const res = await videoApi.result().catch(() => null)
  if (res?.data) {
    if (res.data.video_result) {
      videoResult.value = {
        filename: res.data.video_result.filename,
        video_base64: res.data.video_result.video_base64,
        creations: res.data.video_result.creations,
        prose: res.data.video_result.prose,
      }
    }
    taskId.value = res.data.task_id || ''
    taskState.value = res.data.state || ''
    creations.value = res.data.creations || []
    if (taskState.value && taskState.value !== 'success' && taskState.value !== 'failed') {
      startPolling()
    }
  }
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped>
.video-generation-page {
  max-width: 1500px;
  margin: 0 auto;
}

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

.empty-state {
  margin-top: 60px;
}

.prepare-layout {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}

.prepare-left {
  flex: 1;
  min-width: 0;
}

.prepare-right {
  flex: 1;
  min-width: 0;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.section-header h3 {
  margin: 0;
  font-size: 16px;
  color: #303133;
}

.char-ref-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 520px;
  overflow-y: auto;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  padding: 12px;
  background: #fafafa;
}

.char-ref-row {
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  padding: 10px 12px;
  background: #fff;
}

.char-ref-name {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.char-gid-tag {
  font-size: 12px;
  font-weight: 600;
  color: #409eff;
  background: #ecf5ff;
  padding: 1px 8px;
  border-radius: 4px;
}

.char-name-text {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
}

.char-ref-images {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.char-ref-img-item {
  position: relative;
  width: 72px;
  height: 108px;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #ebeef5;
  background: #fafafa;
}

.char-ref-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.char-view-tag {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  font-size: 10px;
  color: #fff;
  background: rgba(0, 0, 0, 0.55);
  text-align: center;
  padding: 2px 0;
}

.char-no-images {
  font-size: 12px;
  color: #c0c4cc;
  padding: 8px 0;
}

.prepare-next-action {
  display: flex;
  justify-content: center;
  margin-top: 32px;
}

.generate-step {
  display: flex;
  justify-content: center;
  padding-top: 20px;
}

.generate-card {
  width: 720px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 28px 32px;
  background: #fff;
}
.generate-card h3 {
  margin: 0 0 8px 0;
  font-size: 17px;
  color: #303133;
}
.generate-desc {
  margin: 0 0 24px 0;
  font-size: 13px;
  color: #909399;
  line-height: 1.6;
}

.generate-controls {
  display: flex;
  justify-content: center;
  margin-bottom: 24px;
}

.generating-status {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: #909399;
  font-size: 14px;
  margin-bottom: 20px;
}

.video-result {
  margin-top: 20px;
}

.video-player-wrapper {
  border-radius: 8px;
  overflow: hidden;
  background: #000;
}

.video-player {
  width: 100%;
  display: block;
}

.video-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 12px;
}

.generate-error {
  margin-top: 16px;
}

.task-status-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 20px 0;
}

.task-info {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}

.task-id-label {
  color: #909399;
  font-weight: 500;
}

.task-id-value {
  color: #303133;
  font-family: 'Cascadia Code', 'Fira Code', monospace;
  font-size: 12px;
  background: #f5f7fa;
  padding: 4px 10px;
  border-radius: 4px;
}

.task-state {
  display: flex;
  align-items: center;
  gap: 12px;
}

.poll-spinner {
  font-size: 20px;
}

.task-actions {
  display: flex;
  gap: 12px;
}

.task-failed {
  text-align: left;
}

.creations-list {
  margin-top: 12px;
}

.creation-item {
  margin-bottom: 8px;
}

.creation-item a {
  color: #409eff;
  font-size: 13px;
  word-break: break-all;
}

.creation-cover {
  max-width: 200px;
  border-radius: 4px;
  margin-top: 4px;
}
</style>