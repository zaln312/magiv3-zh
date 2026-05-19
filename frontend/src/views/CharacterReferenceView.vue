<template>
  <div class="character-ref-page">
    <div class="page-header">
      <h2>人物参考图</h2>
      <el-button
        type="primary"
        size="small"
        :loading="generatingAll"
        @click="generateAll"
      >
        全部生成
      </el-button>
    </div>

    <div v-if="characters.length === 0" class="empty-state">
      <el-empty description="暂无全局角色，请先在 Predict 页面完成人物检测" />
    </div>

    <div v-else class="reference-workspace">
      <div class="ref-char-list">
        <div
          v-for="char in characters"
          :key="char.global_id"
          class="ref-char-row"
        >
          <div class="char-row-header">
            <span class="char-row-gid">ID: {{ char.global_id }}</span>
            <span class="char-row-name-display">{{ char.name || '未命名' }}</span>
            <el-button
              class="char-gen-btn"
              size="small"
              type="primary"
              :loading="char.generating"
              @click="generateOne(char)"
            >
              生成
            </el-button>
          </div>

          <div class="char-row-body">
            <div class="char-crops-grid">
              <div class="char-crops-section">
                <div class="crops-label">上传参考图</div>
                <div class="uploaded-refs">
                  <div
                    v-for="img in char.uploadedRefImages"
                    :key="img.filename"
                    class="uploaded-ref-item"
                  >
                    <div class="crop-thumb-wrapper">
                      <el-image
                        :src="base64Src(img.image_base64)"
                        :preview-src-list="uploadedPreviewList(char)"
                        :initial-index="uploadedPreviewIndex(char, img.filename)"
                        class="crop-thumb"
                        fit="cover"
                        preview-teleported
                        hide-on-click-modal
                      />
                      <div
                        class="crop-remove-btn"
                        @click.stop="removeUploadedRef(char, img.filename)"
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
                      <el-image
                        :src="base64Src(char.cropImages[cropKey])"
                        :preview-src-list="selectedCropPreviewList(char)"
                        :initial-index="selectedCropPreviewIndex(char, cropKey)"
                        class="crop-thumb"
                        fit="cover"
                        preview-teleported
                        hide-on-click-modal
                      />
                      <div
                        class="crop-deselect-btn"
                        @click.stop="removeCrop(char, cropKey)"
                        title="移回推荐列表"
                      >
                        <span class="deselect-icon">−</span>
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
                  >
                    <div v-if="char.cropImages[cropMeta.crop_key]" class="crop-thumb-wrapper">
                      <el-image
                        :src="base64Src(char.cropImages[cropMeta.crop_key])"
                        :preview-src-list="availableCropPreviewList(char)"
                        :initial-index="availableCropPreviewIndex(char, cropMeta.crop_key)"
                        class="crop-thumb"
                        fit="cover"
                        preview-teleported
                        hide-on-click-modal
                      />
                      <div
                        class="crop-add-btn"
                        title="添加此图片"
                        @click.stop="addCrop(char, cropMeta)"
                      >
                        <span class="add-icon">+</span>
                      </div>
                    </div>
                    <div v-else class="crop-thumb-loading crop-thumb-loading--small">
                      <el-icon class="is-loading"><Loading /></el-icon>
                    </div>
                  </div>
                </div>
              </div>

              </div>

              <div class="char-gen-section">
                <div v-if="char.refImages.length > 0" class="ref-results-gallery">
                  <div
                    v-for="(ref, idx) in char.refImages"
                    :key="idx"
                    class="ref-result-item"
                  >
                    <el-image
                      :src="base64Src(ref.image_base64)"
                      :preview-src-list="refPreviewList(char)"
                      :initial-index="idx"
                      class="ref-img-inline"
                      fit="cover"
                      preview-teleported
                      hide-on-click-modal
                    />
                    <div
                      class="crop-remove-btn"
                      @click.stop="removeRefImage(char, idx)"
                      title="删除此参考图"
                    >
                      <span class="remove-icon">x</span>
                    </div>
                    <div class="ref-mode-tag-inline">{{ viewLabel(ref.view || ref.mode || '') }}</div>
                  </div>
                </div>
                <div v-if="char.refImages.length > 0" class="ref-clear-row">
                  <el-button size="small" type="danger" text @click="clearRefImages(char)">
                    一键清空
                  </el-button>
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

function base64Src(b64: string) {
  return b64 ? `data:image/png;base64,${b64}` : ''
}

function uploadedPreviewList(char: CharacterEntry) {
  return char.uploadedRefImages.map((img) => base64Src(img.image_base64)).filter(Boolean)
}

function uploadedPreviewIndex(char: CharacterEntry, filename: string) {
  return char.uploadedRefImages.findIndex((img) => img.filename === filename)
}

function selectedCropPreviewList(char: CharacterEntry) {
  return char.selectedCropKeys
    .filter((key) => char.cropImages[key])
    .map((key) => base64Src(char.cropImages[key]))
}

function selectedCropPreviewIndex(char: CharacterEntry, cropKey: string) {
  const keys = char.selectedCropKeys.filter((key) => char.cropImages[key])
  return keys.indexOf(cropKey)
}

function availableCropPreviewList(char: CharacterEntry) {
  return availableCrops(char)
    .map((c) => base64Src(char.cropImages[c.crop_key] || ''))
    .filter(Boolean)
}

function availableCropPreviewIndex(char: CharacterEntry, cropKey: string) {
  const crops = availableCrops(char).filter((c) => char.cropImages[c.crop_key])
  return crops.findIndex((c) => c.crop_key === cropKey)
}

function refPreviewList(char: CharacterEntry) {
  return char.refImages.map((ref) => base64Src(ref.image_base64)).filter(Boolean)
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

async function generateOne(char: CharacterEntry) {
  char.generating = true
  try {
    const views = ['front', 'back', 'side']
    const designFilenames = char.uploadedRefImages.map((img) => img.filename)
    const res = await proseApi.generateReferences(
      char.global_id,
      views,
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
    ElMessage.success(
      `角色 "${char.name || 'ID:' + char.global_id}" 参考图生成完成 (${results.length}张)`
    )
  } catch (e: any) {
    ElMessage.error('生成失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    char.generating = false
  }
}

async function generateAll() {
  generatingAll.value = true
  for (const char of characters.value) {
    await generateOne(char)
  }
  generatingAll.value = false
  ElMessage.success('全部参考图生成完成')
}

function clearRefImages(char: CharacterEntry) {
  char.refImages = []
}

function removeRefImage(char: CharacterEntry, idx: number) {
  char.refImages.splice(idx, 1)
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
  margin-right: auto;
}
.empty-state {
  margin-top: 60px;
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

.char-row-name-display {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.char-gen-btn {
  margin-left: auto;
  flex-shrink: 0;
}

.char-row-body {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}

.char-crops-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
  flex: 1;
}

.char-crops-section {
  min-width: 0;
}

.char-gen-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex-shrink: 0;
  align-items: flex-start;
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

.crop-thumb-wrapper {
  width: 100%;
  height: 100%;
  position: relative;
}

.crop-thumb {
  width: 100%;
  height: 100%;
  display: block;
}

.crop-thumb-wrapper :deep(.el-image__inner) {
  width: 100%;
  height: 100%;
}

.crop-remove-btn,
.crop-deselect-btn,
.crop-add-btn {
  z-index: 1;
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

.crop-deselect-btn {
  position: absolute;
  top: 0;
  right: 0;
  width: 18px;
  height: 18px;
  background: rgba(64, 158, 255, 0.9);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  border-radius: 0 0 0 4px;
  opacity: 0;
  transition: opacity 0.15s;
}
.crop-thumb-wrapper:hover .crop-deselect-btn {
  opacity: 1;
}

.deselect-icon {
  font-size: 13px;
  line-height: 1;
  font-weight: 700;
}

.crop-add-btn {
  position: absolute;
  top: 0;
  right: 0;
  width: 18px;
  height: 18px;
  background: rgba(64, 158, 255, 0.9);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 0 0 0 4px;
  opacity: 0;
  transition: opacity 0.15s;
  cursor: pointer;
}
.crop-thumb-wrapper:hover .crop-add-btn {
  opacity: 1;
}

.add-icon {
  font-size: 13px;
  line-height: 1;
  font-weight: 700;
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
  transition: background 0.15s;
  background: #fafafa;
}
.upload-ref-dashed-box:hover {
  background: #f0f2f5;
}

.upload-plus {
  font-size: 24px;
  color: #409eff;
  font-weight: 700;
  line-height: 1;
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
.ref-result-item:hover .crop-remove-btn {
  opacity: 1;
}

.ref-img-inline {
  width: 80px;
  height: 80px;
  display: block;
  border-radius: 4px;
  border: 1px solid #ebeef5;
}

.ref-img-inline :deep(.el-image__inner) {
  width: 80px;
  height: 80px;
  border-radius: 4px;
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

.ref-clear-row {
  margin-top: 4px;
}

.next-step-action {
  display: flex;
  justify-content: center;
  margin-top: 32px;
}
</style>