<template>
  <div class="character-ref-page">
    <div class="page-header">
      <h2>人物参考图</h2>
    </div>

    <div v-if="characters.length === 0" class="empty-state">
      <el-empty description="暂无全局角色，请先在 Predict 页面完成人物检测" />
    </div>

    <div v-else class="reference-workspace">
      <div class="ref-toolbar">
        <span class="ref-title">参考图生成</span>

        <el-checkbox-group v-model="globalViews" size="small" class="view-checkboxes">
          <el-checkbox label="front">正面</el-checkbox>
          <el-checkbox label="back">反面</el-checkbox>
          <el-checkbox label="side">侧面</el-checkbox>
        </el-checkbox-group>

        <el-button
          type="primary"
          size="small"
          :loading="generatingAll"
          @click="generateAll"
        >
          全部生成
        </el-button>
      </div>

      <div class="ref-char-list">
        <div
          v-for="char in characters"
          :key="char.global_id"
          class="ref-char-row"
        >
          <div class="char-row-header">
            <span class="char-row-gid">ID: {{ char.global_id }}</span>
            <el-input
              v-model="char.name"
              size="small"
              placeholder="角色名"
              class="char-row-name"
              @blur="onNameBlur(char)"
              @keyup.enter="($event.target as HTMLElement).blur()"
            />
          </div>

          <div class="char-row-body">
              <div class="char-crops-section">
                <div class="crops-label">上传参考图</div>
                <div class="uploaded-refs">
                  <div
                    v-for="(img, idx) in char.uploadedRefImages"
                    :key="img.filename"
                    class="uploaded-ref-item"
                  >
                    <div class="crop-thumb-wrapper">
                      <img
                        :src="'data:image/png;base64,' + img.image_base64"
                        class="crop-thumb"
                      />
                      <div
                        class="crop-remove-btn"
                        @click="removeUploadedRef(char, img.filename)"
                        title="删除此图片"
                      >
                        <span class="remove-icon">x</span>
                      </div>
                    </div>
                  </div>

                  <div class="upload-ref-trigger" v-if="!char.uploadingRef">
                    <el-upload
                      :auto-upload="false"
                      :show-file-list="false"
                      accept="image/*"
                      @change="(file: any) => onRefUpload(char, file)"
                    >
                      <div class="upload-ref-dashed-box">
                        <span class="upload-plus">+</span>
                      </div>
                    </el-upload>
                  </div>

                  <div v-if="char.uploadingRef" class="crop-thumb-loading">
                    <el-icon class="is-loading"><Loading /></el-icon>
                  </div>
                </div>
              </div>

              <div class="char-crops-section">
                <div class="crops-label">选出的人物图片</div>
                <div class="selected-crops">
                  <div
                    v-for="cropKey in char.selectedCropKeys"
                    :key="cropKey"
                    class="selected-crop-item"
                  >
                    <div v-if="char.cropImages[cropKey]" class="crop-thumb-wrapper">
                      <img
                        :src="'data:image/png;base64,' + char.cropImages[cropKey]"
                        class="crop-thumb"
                      />
                      <div
                        class="crop-remove-btn"
                        @click="removeCrop(char, cropKey)"
                        title="移除此图片"
                      >
                        <span class="remove-icon">x</span>
                      </div>
                    </div>
                    <div v-else class="crop-thumb-loading">
                      <el-icon class="is-loading"><Loading /></el-icon>
                    </div>
                  </div>
                  <div v-if="char.selectedCropKeys.length === 0" class="crop-thumb-placeholder">
                    未选择
                  </div>
                </div>
              </div>

              <div class="char-crops-section" v-if="availableCrops(char).length > 0">
                <div class="crops-label">推荐截取列表</div>
                <div class="available-crops">
                  <div
                    v-for="cropMeta in availableCrops(char)"
                    :key="cropMeta.crop_key"
                    class="available-crop-item"
                    @click="addCrop(char, cropMeta)"
                    title="添加此图片"
                  >
                    <div v-if="char.cropImages[cropMeta.crop_key]" class="crop-thumb-wrapper">
                      <img
                        :src="'data:image/png;base64,' + char.cropImages[cropMeta.crop_key]"
                        class="crop-thumb crop-thumb--candidate"
                      />
                      <div class="crop-add-overlay">
                        <span class="add-icon">+</span>
                      </div>
                    </div>
                    <div v-else class="crop-thumb-loading crop-thumb-loading--small">
                      <el-icon class="is-loading"><Loading /></el-icon>
                    </div>
                    <div class="crop-priority-badge">{{ cropMeta.rank }}</div>
                  </div>
                </div>
              </div>

              <div class="char-gen-section">
                <div class="gen-controls">
                  <el-checkbox-group v-model="char.selectedViews" size="small" class="view-checkboxes">
                    <el-checkbox label="front">正</el-checkbox>
                    <el-checkbox label="back">反</el-checkbox>
                    <el-checkbox label="side">侧</el-checkbox>
                  </el-checkbox-group>
                  <el-button
                    size="small"
                    type="primary"
                    :loading="char.generating"
                    @click="generateOne(char)"
                  >
                    生成
                  </el-button>
                </div>

                <div v-if="char.refImages.length > 0" class="ref-results-gallery">
                  <div
                    v-for="(ref, idx) in char.refImages"
                    :key="idx"
                    class="ref-result-item"
                  >
                    <img
                      :src="'data:image/png;base64,' + ref.image_base64"
                      class="ref-img-inline"
                    />
                    <div class="ref-mode-tag-inline">{{ viewLabel(ref.view || ref.mode || '') }}</div>
                  </div>
                </div>
                <div v-if="char.refImages.length > 0" class="ref-clear-row">
                  <el-button size="small" type="danger" text @click="clearRefImages(char)">
                    一键清空
                  </el-button>
                </div>
                <div v-else-if="char.generating" class="ref-generating-inline">
                  <el-icon class="is-loading"><Loading /></el-icon>
                  <span>生成中...</span>
                </div>
              </div>
            </div>
        </div>
      </div>
    </div>

    <div v-if="characters.length > 0" class="next-step-action">
      <el-button type="primary" size="large" @click="goVideoGeneration">
        下一步：视频生成
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Loading } from '@element-plus/icons-vue'
import { proseApi, characterApi } from '../api/endpoints'

const router = useRouter()

const generatingAll = ref(false)
const globalViews = ref<string[]>(['front', 'back', 'side'])

interface CropMeta {
  crop_key: string
  img_idx: number
  char_idx: number
  box: number[]
  area: number
  rank: number
}

interface RefImage {
  image_base64: string
  view?: string
  mode?: string
  prompt?: string
}

interface DesignImage {
  filename: string
  image_base64: string
}

interface CharacterEntry {
  global_id: number
  name: string
  generating: boolean
  uploadingRef: boolean
  allCrops: CropMeta[]
  selectedCropKeys: string[]
  cropImages: Record<string, string>
  refImages: RefImage[]
  selectedViews: string[]
  uploadedRefImages: DesignImage[]
}

const characters = ref<CharacterEntry[]>([])

const VIEW_LABELS: Record<string, string> = {
  front: '正面',
  back: '反面',
  side: '侧面',
}

function viewLabel(v: string) {
  return VIEW_LABELS[v] || v
}

async function loadCharacters() {
  try {
    const refRes = await proseApi.referenceResults()
    const lib: { global_id: number; name: string }[] = refRes.data.character_library || []
    const existingRefs: Record<string, any> = refRes.data.character_references || {}

    characters.value = lib.map((entry) => {
      const existing = existingRefs[String(entry.global_id)]
      const refs: RefImage[] = []
      if (existing?.references) {
        for (const r of existing.references) {
          refs.push({
            image_base64: r.image_base64 || '',
            view: r.view || r.mode || '',
            mode: r.mode || '',
            prompt: r.prompt || '',
          })
        }
      }

      return {
        global_id: entry.global_id,
        name: entry.name || '',
        generating: false,
        uploadingRef: false,
        allCrops: [],
        selectedCropKeys: [],
        cropImages: {},
        refImages: refs,
        selectedViews: ['front', 'back', 'side'],
        uploadedRefImages: [],
      } as CharacterEntry
    })

    for (const char of characters.value) {
      loadAllCrops(char)
      loadUploadedRefs(char)
    }
  } catch {}
}

async function loadAllCrops(char: CharacterEntry) {
  try {
    const res = await proseApi.characterAllCrops(char.global_id)
    const crops: any[] = res.data.crops || []
    char.allCrops = crops.map((c: any, idx: number) => ({
      ...c,
      rank: idx + 1,
    }))

    if (char.allCrops.length > 0) {
      char.selectedCropKeys = [char.allCrops[0].crop_key]
      for (const crop of char.allCrops) {
        fetchCropImage(char, crop.crop_key)
      }
    }
  } catch {
    char.allCrops = []
    char.selectedCropKeys = []
  }
}

async function fetchCropImage(char: CharacterEntry, cropKey: string) {
  try {
    const res = await proseApi.characterCropImage(char.global_id, cropKey)
    char.cropImages[cropKey] = res.data.image_base64 || ''
  } catch {
    char.cropImages[cropKey] = ''
  }
}

function availableCrops(char: CharacterEntry): CropMeta[] {
  const selectedSet = new Set(char.selectedCropKeys)
  return char.allCrops.filter((c) => !selectedSet.has(c.crop_key))
}

function addCrop(char: CharacterEntry, cropMeta: CropMeta) {
  if (!char.selectedCropKeys.includes(cropMeta.crop_key)) {
    char.selectedCropKeys.push(cropMeta.crop_key)
    if (!char.cropImages[cropMeta.crop_key]) {
      fetchCropImage(char, cropMeta.crop_key)
    }
  }
}

function removeCrop(char: CharacterEntry, cropKey: string) {
  char.selectedCropKeys = char.selectedCropKeys.filter((k) => k !== cropKey)
}

async function onNameBlur(char: CharacterEntry) {
  try {
    await characterApi.updateName(char.global_id, char.name)
  } catch {
    ElMessage.error('角色名更新失败')
  }
}

async function generateOne(char: CharacterEntry) {
  if (char.selectedViews.length === 0) {
    ElMessage.warning('请至少选择一个视图（正/反/侧）')
    return
  }
  char.generating = true
  try {
    const designFilenames = char.uploadedRefImages.map((img) => img.filename)
    const res = await proseApi.generateReferences(
      char.global_id,
      char.selectedViews,
      char.selectedCropKeys,
      1,
      designFilenames,
    )
    const results = res.data.results || []
    for (const r of results) {
      char.refImages.push({
        image_base64: r.image_base64 || '',
        view: r.view || '',
        prompt: r.prompt || '',
      })
    }
    const viewsStr = char.selectedViews.map(v => viewLabel(v)).join('/')
    ElMessage.success(
      `角色 "${char.name || 'ID:' + char.global_id}" ${viewsStr}视图生成完成 (${results.length}张)`
    )
  } catch (e: any) {
    ElMessage.error('生成失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    char.generating = false
  }
}

async function generateAll() {
  if (globalViews.value.length === 0) {
    ElMessage.warning('请至少选择一个视图（正/反/侧）')
    return
  }
  generatingAll.value = true
  for (const char of characters.value) {
    char.selectedViews = [...globalViews.value]
    await generateOne(char)
  }
  generatingAll.value = false
  ElMessage.success('全部参考图生成完成')
}

function clearRefImages(char: CharacterEntry) {
  char.refImages = []
}

async function loadUploadedRefs(char: CharacterEntry) {
  try {
    const res = await characterApi.designImages(char.global_id)
    char.uploadedRefImages = res.data.images || []
  } catch {
    char.uploadedRefImages = []
  }
}

async function onRefUpload(char: CharacterEntry, file: any) {
  char.uploadingRef = true
  try {
    const res = await characterApi.uploadDesignImage(char.global_id, file.raw)
    if (res.data.success) {
      await loadUploadedRefs(char)
      ElMessage.success('参考图上传成功')
    }
  } catch (e: any) {
    ElMessage.error('上传失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    char.uploadingRef = false
  }
}

async function removeUploadedRef(char: CharacterEntry, filename: string) {
  try {
    await characterApi.deleteDesignImage(char.global_id, filename)
    char.uploadedRefImages = char.uploadedRefImages.filter((img) => img.filename !== filename)
    ElMessage.success('已删除')
  } catch (e: any) {
    ElMessage.error('删除失败: ' + (e.response?.data?.detail || e.message))
  }
}

function goVideoGeneration() {
  router.push('/video-generation')
}

onMounted(() => {
  loadCharacters()
})
</script>

<style scoped>
.character-ref-page {
  max-width: 1100px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 24px;
}
.page-header h2 {
  margin: 0;
  font-size: 24px;
}
.empty-state {
  margin-top: 60px;
}

.reference-workspace {}

.ref-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
  padding: 10px 16px;
  background: #f5f7fa;
  border-radius: 8px;
  flex-wrap: wrap;
}
.ref-title {
  font-size: 15px;
  font-weight: 600;
  color: #303133;
  margin-right: auto;
}

.view-checkboxes {
  display: flex;
  gap: 0;
}

.ref-char-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.ref-char-row {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 16px;
  background: #fff;
}

.char-row-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid #ebeef5;
}

.char-row-gid {
  font-size: 13px;
  font-weight: 600;
  color: #409eff;
  background: #ecf5ff;
  padding: 2px 10px;
  border-radius: 4px;
  white-space: nowrap;
}

.char-row-name {
  width: 180px;
}

.char-row-body {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}

.char-crops-section {
  flex: 1;
  min-width: 0;
}

.char-gen-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex-shrink: 0;
  align-items: flex-start;
}

.gen-controls {
  display: flex;
  align-items: center;
  gap: 6px;
}

.crops-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 8px;
  font-weight: 500;
}

.selected-crops,
.available-crops {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.selected-crop-item,
.available-crop-item {
  position: relative;
  width: 64px;
  height: 64px;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #ebeef5;
  background: #fafafa;
}

.available-crop-item {
  cursor: pointer;
  transition: border-color 0.2s;
}
.available-crop-item:hover {
  border-color: #409eff;
}

.crop-thumb-wrapper {
  width: 100%;
  height: 100%;
  position: relative;
}

.crop-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.crop-thumb--candidate {
  filter: brightness(0.95);
}

.crop-remove-btn {
  position: absolute;
  top: 0;
  right: 0;
  width: 18px;
  height: 18px;
  background: rgba(245, 108, 108, 0.9);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  border-radius: 0 0 0 4px;
  opacity: 0;
  transition: opacity 0.15s;
}
.crop-thumb-wrapper:hover .crop-remove-btn {
  opacity: 1;
}

.remove-icon {
  font-size: 13px;
  line-height: 1;
  font-weight: 700;
}

.crop-add-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(64, 158, 255, 0.15);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.15s;
}
.available-crop-item:hover .crop-add-overlay {
  opacity: 1;
}

.add-icon {
  font-size: 24px;
  color: #409eff;
  font-weight: 700;
  line-height: 1;
}

.crop-priority-badge {
  position: absolute;
  top: 0;
  left: 0;
  background: rgba(64, 158, 255, 0.85);
  color: #fff;
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 0 0 4px 0;
  font-weight: 600;
  line-height: 1.4;
}

.crop-thumb-loading {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #c0c4cc;
}

.crop-thumb-loading--small {
  font-size: 16px;
}

.crop-thumb-placeholder {
  font-size: 12px;
  color: #c0c4cc;
  padding: 8px 12px;
  border: 1px dashed #dcdfe6;
  border-radius: 4px;
}

.uploaded-refs {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.uploaded-ref-item {
  position: relative;
  width: 64px;
  height: 64px;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #ebeef5;
  background: #fafafa;
}

.upload-ref-trigger {
  width: 64px;
  height: 64px;
}

.upload-ref-dashed-box {
  width: 64px;
  height: 64px;
  border: 1px dashed #dcdfe6;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: border-color 0.2s;
  background: #fafafa;
}
.upload-ref-dashed-box:hover {
  border-color: #409eff;
}

.upload-plus {
  font-size: 28px;
  color: #c0c4cc;
  font-weight: 300;
}

.ref-results-gallery {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.ref-result-item {
  position: relative;
  flex-shrink: 0;
}

.ref-img-inline {
  width: 80px;
  height: 80px;
  object-fit: cover;
  border-radius: 4px;
  border: 1px solid #ebeef5;
}

.ref-mode-tag-inline {
  position: absolute;
  top: 1px;
  right: 1px;
  background: rgba(64, 158, 255, 0.9);
  color: #fff;
  font-size: 9px;
  padding: 1px 5px;
  border-radius: 3px;
}

.ref-generating-inline {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #409eff;
  font-size: 13px;
  padding: 8px 14px;
  background: #ecf5ff;
  border-radius: 4px;
}

.ref-clear-row {
  margin-top: 4px;
}

.next-step-action {
  display: flex;
  justify-content: center;
  margin-top: 32px;
}
</style>