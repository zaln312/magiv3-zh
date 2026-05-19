<template>
  <div class="home-page">
    <div class="page-header">
      <h2>Magi Studio</h2>
      <div class="header-right">
        <div class="magi-mode-info" v-if="currentMagiMode">
          <span class="label">Magi V3 模式：</span>
          <el-tag :type="currentMagiMode === 'dynamic' ? 'info' : 'primary'" size="small">
            {{ magiModeLabel[currentMagiMode] }}
          </el-tag>
        </div>
        <el-button type="primary" size="large" @click="showCreateDialog = true">
          <el-icon><Plus /></el-icon>
          新建项目
        </el-button>
      </div>
    </div>

    <div v-if="loading" class="loading-area">
      <el-icon class="is-loading" :size="32"><Loading /></el-icon>
      <span>加载中...</span>
    </div>

    <div v-else-if="projects.length === 0" class="empty-area">
      <el-empty description="暂无项目，点击上方按钮创建第一个项目" />
    </div>

    <div v-else class="project-grid">
      <el-card
        v-for="p in projects"
        :key="p.id"
        class="project-card"
        shadow="hover"
        @click="enterProject(p)"
      >
        <div class="card-content">
          <div class="card-name">{{ p.name || '未命名项目' }}</div>
          <div class="card-meta">
            <el-tag
              :type="stepTagType(p.current_step)"
              size="small"
            >
              {{ stepLabel(p.current_step) }}
            </el-tag>
            <el-tag
              v-if="p.video_task_state && p.video_task_state !== 'success' && p.video_task_state !== 'failed'"
              :type="(videoStateTags[p.video_task_state] || 'info') as any"
              size="small"
            >
              {{ videoStateLabels[p.video_task_state] || '视频生成中' }}
            </el-tag>
            <span class="card-date">{{ formatDate(p.updated_at) }}</span>
          </div>
        </div>
        <div class="card-actions" @click.stop>
          <el-button text size="small" @click="startRename(p)">
            <el-icon><Edit /></el-icon>
          </el-button>
          <el-popconfirm
            title="确定要删除此项目吗？此操作不可撤销。"
            @confirm="handleDelete(p.id)"
          >
            <template #reference>
              <el-button text size="small" type="danger">
                <el-icon><Delete /></el-icon>
              </el-button>
            </template>
          </el-popconfirm>
        </div>
      </el-card>
    </div>

    <el-dialog v-model="showCreateDialog" title="新建项目" width="420px" :close-on-click-modal="false">
      <el-form @submit.prevent="handleCreate">
        <el-form-item label="项目名称">
          <el-input
            v-model="newProjectName"
            placeholder="输入项目名称（可选）"
            maxlength="50"
            clearable
            @keyup.enter="handleCreate"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="handleCreate" :loading="creating">创建</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="showRenameDialog" title="重命名项目" width="360px">
      <el-input
        v-model="renameText"
        placeholder="输入新名称"
        maxlength="50"
        @keyup.enter="handleRename"
      />
      <template #footer>
        <el-button @click="showRenameDialog = false">取消</el-button>
        <el-button type="primary" @click="handleRename">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Plus, Edit, Delete, Loading } from '@element-plus/icons-vue'
import { projectApi, configApi } from '../api/endpoints'

interface Project {
  id: string
  name: string
  current_step: string
  created_at: string
  updated_at: string
  data_exists: boolean
  video_task_id: string | null
  video_task_state: string
  video_creations: any[]
}

const router = useRouter()

const projects = ref<Project[]>([])
const loading = ref(true)
const creating = ref(false)
const showCreateDialog = ref(false)
const newProjectName = ref('')

const showRenameDialog = ref(false)
const renameTarget = ref<Project | null>(null)
const renameText = ref('')

const currentMagiMode = ref<string>('')
const magiModeLabel: Record<string, string> = {
  dynamic: '动态加载卸载',
  persistent_project: '进入项目保持加载',
}

const stepLabels: Record<string, string> = {
  upload: '上传图片',
  ocr: 'OCR 识别',
  predict: 'Predict 预测',
  caption: 'Caption 描述',
  grounding: 'Grounding 指代',
  prose: 'Prose 叙述',
  video: '视频生成',
}

const stepTags: Record<string, string> = {
  upload: 'info',
  ocr: 'warning',
  predict: '',
  caption: 'success',
  grounding: 'success',
  prose: 'success',
  video: 'warning',
}

const videoStateLabels: Record<string, string> = {
  created: '视频创建',
  queueing: '视频排队',
  processing: '视频处理中',
  success: '视频完成',
  failed: '视频失败',
}

const videoStateTags: Record<string, string> = {
  created: 'info',
  queueing: 'warning',
  processing: '',
  success: 'success',
  failed: 'danger',
}

function stepLabel(step: string): string {
  return stepLabels[step] || step
}

function stepTagType(step: string): '' | 'success' | 'warning' | 'info' | 'danger' {
  return (stepTags[step] || 'info') as '' | 'success' | 'warning' | 'info' | 'danger'
}

function formatDate(dateStr: string): string {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

async function loadProjects() {
  loading.value = true
  try {
    const res = await projectApi.list()
    projects.value = res.data.projects || []
  } catch {
    ElMessage.error('加载项目列表失败')
  } finally {
    loading.value = false
  }
}

async function handleCreate() {
  creating.value = true
  try {
    const res = await projectApi.create(newProjectName.value)
    ElMessage.success('项目创建成功')
    showCreateDialog.value = false
    newProjectName.value = ''
    const project = res.data.project
    router.push(`/upload?project_id=${project.id}`)
  } catch (e: any) {
    ElMessage.error('创建失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    creating.value = false
  }
}

async function enterProject(project: Project) {
  try {
    await projectApi.load(project.id)
    router.push(`/upload?project_id=${project.id}`)
  } catch (e: any) {
    ElMessage.error('加载项目失败: ' + (e.response?.data?.detail || e.message))
  }
}

function startRename(project: Project) {
  renameTarget.value = project
  renameText.value = project.name
  showRenameDialog.value = true
}

async function handleRename() {
  if (!renameTarget.value) return
  try {
    await projectApi.rename(renameTarget.value.id, renameText.value)
    ElMessage.success('已重命名')
    renameTarget.value.name = renameText.value
    showRenameDialog.value = false
  } catch (e: any) {
    ElMessage.error('重命名失败')
  }
}

async function handleDelete(projectId: string) {
  try {
    await projectApi.delete(projectId)
    projects.value = projects.value.filter((p) => p.id !== projectId)
    ElMessage.success('已删除')
  } catch (e: any) {
    ElMessage.error('删除失败')
  }
}

async function loadMagiMode() {
  try {
    const res = await configApi.getMagiMode()
    if (res.data.success) {
      currentMagiMode.value = res.data.mode
    }
  } catch {
  }
}

onMounted(() => {
  loadProjects()
  loadMagiMode()
})
</script>

<style scoped>
.home-page {
  max-width: 1000px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
}

.page-header h2 {
  margin: 0;
  font-size: 24px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.magi-mode-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.magi-mode-info .label {
  color: #606266;
  font-size: 14px;
}

.loading-area {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 80px 0;
  color: #909399;
}

.empty-area {
  padding: 60px 0;
}

.project-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}

.project-card {
  cursor: pointer;
  transition: transform 0.2s;
}

.project-card:hover {
  transform: translateY(-2px);
}

.project-card :deep(.el-card__body) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
}

.card-content {
  flex: 1;
  min-width: 0;
}

.card-name {
  font-size: 16px;
  font-weight: 500;
  margin-bottom: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.card-date {
  font-size: 12px;
  color: #909399;
}

.card-actions {
  display: flex;
  gap: 4px;
  margin-left: 8px;
}
</style>