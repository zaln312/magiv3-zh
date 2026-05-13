<template>
  <div class="prose-page">
    <div class="page-header">
      <h2>Prose 生成</h2>
    </div>

    <div v-if="!prosePrompt && !proseText" class="empty-state">
      <el-empty description="暂无 Prose 结果，请先执行 Prose" />
    </div>

    <div v-else class="prose-content">
      <div class="prose-section">
        <h3>Prose Prompt</h3>
        <el-input
          v-model="prosePrompt"
          type="textarea"
          :rows="16"
          resize="vertical"
          @change="updatePrompt"
        />
      </div>

      <div class="prose-section">
        <h3>Prose 文本</h3>
        <el-input
          v-model="proseText"
          type="textarea"
          :rows="20"
          resize="vertical"
          @change="updateText"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { proseApi } from '../api/endpoints'

const prosePrompt = ref('')
const proseText = ref('')

async function loadProse() {
  try {
    const res = await proseApi.results()
    prosePrompt.value = res.data.prose_prompt || ''
    proseText.value = res.data.prose_text || ''
  } catch {}
}

async function updatePrompt() {
  try { await proseApi.updatePrompt(prosePrompt.value) } catch { ElMessage.error('更新失败') }
}

async function updateText() {
  try { await proseApi.updateText(proseText.value) } catch { ElMessage.error('更新失败') }
}

onMounted(loadProse)
</script>

<style scoped>
.prose-page { max-width: 1000px; margin: 0 auto; }
.page-header { margin-bottom: 24px; }
.page-header h2 { margin: 0; }
.empty-state { margin-top: 60px; }

.prose-content { display: flex; flex-direction: column; gap: 24px; }
.prose-section h3 { margin: 0 0 10px 0; font-size: 15px; }
</style>