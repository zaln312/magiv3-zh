<template>
  <div class="prose-prompt-page">
    <div class="page-header">
      <h2>Prose Prompt</h2>
      <el-button type="primary" @click="runProse" :loading="proseLoading">
        生成 Prose
      </el-button>
    </div>

    <div v-if="!prosePrompt" class="empty-state">
      <el-empty description="暂无 Prose Prompt，请先从 Grounding 页面构建" />
    </div>

    <div v-else class="prompt-content">
      <el-input
        v-model="prosePrompt"
        type="textarea"
        :rows="24"
        resize="vertical"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { proseApi } from '../api/endpoints'

const router = useRouter()

const prosePrompt = ref('')
const proseLoading = ref(false)

async function loadPrompt() {
  try {
    const res = await proseApi.results()
    prosePrompt.value = res.data.prose_prompt || ''
  } catch {}
}

async function runProse() {
  proseLoading.value = true
  try {
    await proseApi.run()
    ElMessage.success('Prose 生成完成')
    router.push('/prose')
  } catch (e: any) {
    ElMessage.error('Prose 生成失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    proseLoading.value = false
  }
}

onMounted(loadPrompt)
</script>

<style scoped>
.prose-prompt-page { max-width: 1000px; margin: 0 auto; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
.page-header h2 { margin: 0; }
.empty-state { margin-top: 60px; }
.prompt-content { border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; }
</style>