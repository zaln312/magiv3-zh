<template>
  <div class="prose-page">
    <div class="sub-nav">
      <div
        class="sub-nav-item"
        :class="{ active: currentStep === 0 }"
        @click="currentStep = 0"
      >
        Prose Prompt
      </div>
      <div
        class="sub-nav-item"
        :class="{ active: currentStep === 1 }"
        @click="currentStep = 1"
      >
        Prose 叙述
      </div>
    </div>

    <div v-show="currentStep === 0" class="prompt-step">
      <div class="prompt-card">
        <h3>Prose Prompt 设置</h3>
        <p class="prompt-desc">
          编辑 Prose 生成的 Prompt 文本，用于指导语言模型生成最终的叙述内容。可用其他编辑器编辑好后粘贴替换。
        </p>

        <div class="prompt-form">
          <el-input
            v-model="prosePromptText"
            type="textarea"
            :rows="16"
            resize="vertical"
            placeholder="暂无 Prompt，请先在 Grounding 页面执行生成 Prose…"
          />
        </div>

        <div class="prompt-actions">
          <el-button type="primary" @click="goNext" :loading="proseLoading">
            执行
          </el-button>
        </div>
      </div>
    </div>

    <div v-show="currentStep === 1">
      <div class="page-header">
        <h2>Prose 叙述</h2>
      </div>

      <div v-if="!proseText" class="empty-state">
        <el-empty description="暂无 Prose 结果，请先执行 Prose 生成" />
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
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { proseApi } from '../api/endpoints'

const currentStep = ref(0)
const proseText = ref('')
const prosePromptText = ref('')
const proseLoading = ref(false)

async function loadProse() {
  try {
    const res = await proseApi.results()
    proseText.value = res.data.prose || ''
    const promptArr = res.data.prose_prompt || []
    prosePromptText.value = promptArr.join('\n')
  } catch {}
}

async function goNext() {
  proseLoading.value = true
  try {
    await proseApi.run()
    ElMessage.success('Prose 生成完成')
    currentStep.value = 1
    await loadProse()
  } catch (e: any) {
    ElMessage.error('Prose 生成失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    proseLoading.value = false
  }
}

onMounted(loadProse)
</script>

<style scoped>
.prose-page { max-width: 1000px; margin: 0 auto; }

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

.prompt-step {
  display: flex;
  justify-content: center;
  padding-top: 20px;
}
.prompt-card {
  width: 720px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 28px 32px;
  background: #fff;
}
.prompt-card h3 {
  margin: 0 0 8px 0;
  font-size: 17px;
  color: #303133;
}
.prompt-desc {
  margin: 0 0 24px 0;
  font-size: 13px;
  color: #909399;
  line-height: 1.6;
}
.prompt-form {
  margin-bottom: 24px;
}
.prompt-actions {
  display: flex;
  justify-content: flex-end;
}

.page-header { margin-bottom: 24px; }
.page-header h2 { margin: 0; }
.empty-state { margin-top: 60px; }
.prose-content { border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; }
</style>