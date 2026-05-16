<template>
  <div class="upload-page">
    <div class="page-header">
      <h2>上传图片</h2>
      <el-button type="primary" @click="runOcr" :disabled="imgPaths.length === 0" :loading="ocrLoading">
        执行 OCR 识别
      </el-button>
    </div>

    <el-upload
      v-if="imgPaths.length === 0"
      class="upload-area"
      drag
      multiple
      accept="image/*"
      :auto-upload="false"
      :show-file-list="false"
      :on-change="handleFilesAdded"
    >
      <el-icon class="el-icon--upload"><upload-filled /></el-icon>
      <div class="el-upload__text">
        将图片拖到此处，或<span>点击上传</span>
      </div>
    </el-upload>

    <div v-else class="image-grid-wrapper">
      <div class="image-grid">
        <draggable
          v-model="imgPaths"
          :animation="200"
          :item-key="(item: string) => item"
          ghost-class="image-card--ghost"
          chosen-class="image-card--chosen"
          drag-class="image-card--drag"
          @start="onDragStart"
          @end="onDragEnd"
          tag="div"
          class="draggable-inner"
        >
          <template #item="{ element: path, index: idx }">
            <div class="image-card">
              <div class="image-index">{{ idx + 1 }}</div>
              <img :src="serveUrl(idx)" class="preview-img" />
              <div class="file-name">{{ getFileName(idx) }}</div>
              <el-button
                class="delete-btn"
                type="danger"
                circle
                size="small"
                :icon="Delete"
                @click="deleteImage(path)"
              />
            </div>
          </template>
        </draggable>

        <div
          class="image-card add-card"
          :class="{ 'add-card--hover': dragOverAdd }"
          @click="triggerUpload"
          @dragover.prevent="onAddCardDragOver"
          @dragleave="onAddCardDragLeave"
          @drop="onAddCardDrop"
        >
          <el-icon :size="40"><plus /></el-icon>
          <span>添加图片</span>
        </div>
      </div>
    </div>

    <el-upload
      ref="hiddenUploadRef"
      class="hidden-upload"
      multiple
      accept="image/*"
      :auto-upload="false"
      :show-file-list="false"
      :on-change="handleFilesAdded"
    >
      <template #trigger>
        <span ref="triggerSpan" style="display:none"></span>
      </template>
    </el-upload>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { UploadFilled, Plus, Delete } from '@element-plus/icons-vue'
import draggable from 'vuedraggable'
import { uploadApi, ocrApi } from '../api/endpoints'

const router = useRouter()

const imgPaths = ref<string[]>([])
const fileNames = ref<Record<string, string>>({})
const ocrLoading = ref(false)
const hiddenUploadRef = ref<any>(null)
const pendingFiles = ref<File[]>([])
let flushTimer: ReturnType<typeof setTimeout> | null = null

const oldImgPaths = ref<string[]>([])
const dragOverAdd = ref(false)

function serveUrl(idx: number) {
  const path = imgPaths.value[idx]
  if (!path) return ''
  const filename = path.split('/').pop() || ''
  return uploadApi.serveByNameUrl(filename)
}

function getFileName(idx: number) {
  const path = imgPaths.value[idx]
  if (!path) return ''
  return fileNames.value[path] || path.split('/').pop() || ''
}

async function loadImages() {
  try {
    const res = await uploadApi.list()
    imgPaths.value = res.data.img_paths
  } catch {
    imgPaths.value = []
  }
}

function handleFilesAdded(file: any) {
  const raw = file.raw
  if (!raw) return
  pendingFiles.value.push(raw)
  if (flushTimer) clearTimeout(flushTimer)
  flushTimer = setTimeout(() => flushPendingFiles(), 50)
}

async function flushPendingFiles() {
  const files = pendingFiles.value.splice(0)
  if (files.length === 0) return
  const oldLen = imgPaths.value.length
  try {
    const res = await uploadApi.upload(files)
    imgPaths.value = res.data.img_paths
    for (let i = 0; i < files.length; i++) {
      const newPath = imgPaths.value[oldLen + i]
      if (newPath) {
        fileNames.value[newPath] = files[i].name
      }
    }
    ElMessage.success('已上传')
  } catch (e: any) {
    ElMessage.error('上传失败: ' + (e.response?.data?.detail || e.message))
  }
}

function triggerUpload() {
  const span = document.querySelector('.hidden-upload .el-upload span') as HTMLElement
  span?.click()
}

function onAddCardDragOver() {
  dragOverAdd.value = true
}

function onAddCardDragLeave() {
  dragOverAdd.value = false
}

function onAddCardDrop(e: DragEvent) {
  e.preventDefault()
  dragOverAdd.value = false
  const files = e.dataTransfer?.files
  if (!files || files.length === 0) return
  for (let i = 0; i < files.length; i++) {
    handleFilesAdded({ raw: files[i] })
  }
}

async function deleteImage(path: string) {
  try {
    const res = await uploadApi.deleteImage(path)
    imgPaths.value = res.data.img_paths
    delete fileNames.value[path]
    ElMessage.success('已删除')
  } catch (e: any) {
    ElMessage.error('删除失败')
  }
}

function onDragStart() {
  oldImgPaths.value = [...imgPaths.value]
}

async function onDragEnd() {
  if (oldImgPaths.value.length === 0) return

  const newOrder = imgPaths.value.map((p) => oldImgPaths.value.indexOf(p))
  const isChanged = newOrder.some((val, idx) => val !== idx)
  if (!isChanged) {
    oldImgPaths.value = []
    return
  }

  try {
    const res = await uploadApi.reorder(newOrder)
    imgPaths.value = res.data.img_paths
  } catch {
    imgPaths.value = oldImgPaths.value
    ElMessage.error('排序失败')
  }
  oldImgPaths.value = []
}

async function runOcr() {
  ocrLoading.value = true
  try {
    await ocrApi.run()
    ElMessage.success('OCR 识别完成')
    router.push('/ocr')
  } catch (e: any) {
    ElMessage.error('OCR 失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    ocrLoading.value = false
  }
}

onMounted(loadImages)
</script>

<style scoped>
.upload-page {
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0;
}

.upload-area {
  width: 100%;
}

.image-grid-wrapper {
  width: 100%;
}

.image-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

.draggable-inner {
  display: contents;
}

.image-card {
  position: relative;
  width: 200px;
  height: 310px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
  cursor: grab;
  transition: border-color 0.2s, transform 0.2s;
  display: flex;
  flex-direction: column;
}

.image-card:hover {
  border-color: #409eff;
}

.image-card--ghost {
  opacity: 0.4;
  border-style: dashed;
  border-color: #409eff;
  background: #ecf5ff;
}

.image-card--chosen {
  opacity: 0.5;
}

.image-card--drag {
  opacity: 0.9;
  transform: rotate(3deg);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
}

.image-index {
  position: absolute;
  top: 4px;
  left: 4px;
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  z-index: 2;
}

.preview-img {
  flex: 1;
  width: 100%;
  object-fit: cover;
  min-height: 0;
}

.delete-btn {
  position: absolute;
  top: 4px;
  right: 4px;
  z-index: 2;
}

.file-name {
  padding: 4px 8px;
  font-size: 12px;
  color: #666;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  background: #fafafa;
  border-top: 1px solid #eee;
  flex-shrink: 0;
}

.add-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: #999;
  cursor: pointer;
  border-style: dashed;
}

.add-card:hover,
.add-card--hover {
  color: #409eff;
  border-color: #409eff;
  background: #ecf5ff;
}

.hidden-upload {
  display: none;
}
</style>