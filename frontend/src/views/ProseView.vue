<template>
  <div class="prose-page">
    <div class="page-header">
      <h2>Prose 叙述</h2>
    </div>

    <div v-if="!proseText" class="empty-state">
      <el-empty description="暂无 Prose 结果，请先从 Prose Prompt 页面生成" />
    </div>

    <div v-else class="prose-content">
      <el-input
        v-model="proseText"
        type="textarea"
        :rows="24"
        resize="vertical"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { proseApi } from '../api/endpoints'

const proseText = ref('')

async function loadProse() {
  try {
    const res = await proseApi.results()
    proseText.value = res.data.prose || ''
  } catch {}
}

onMounted(loadProse)
</script>

<style scoped>
.prose-page { max-width: 1000px; margin: 0 auto; }
.page-header { margin-bottom: 24px; }
.page-header h2 { margin: 0; }
.empty-state { margin-top: 60px; }
.prose-content { border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; }
</style>