<template>
  <div id="app">
    <el-container>
      <el-header>
        <el-menu
            :default-active="currentRoute"
            mode="horizontal"
            :ellipsis="false"
            router
          >
          <el-menu-item index="/">首页</el-menu-item>
          <el-menu-item index="/settings">系统设置</el-menu-item>
          <el-menu-item index="/upload">上传图片</el-menu-item>
          <el-menu-item index="/ocr">OCR 识别</el-menu-item>
          <el-menu-item index="/predict">Predict 预测</el-menu-item>
          <el-menu-item index="/grounding">Grounding 指代</el-menu-item>
          <el-menu-item index="/prose">Prose 叙述</el-menu-item>
          <el-menu-item index="/character-reference">人物参考图</el-menu-item>
          <el-menu-item index="/video-generation">视频生成</el-menu-item>
        </el-menu>
        <div class="project-badge" v-if="projectName">
          <el-tag type="primary" size="small">{{ projectName }}</el-tag>
        </div>
      </el-header>
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { projectApi } from './api/endpoints'

const route = useRoute()
const currentRoute = computed(() => route.path)
const projectName = ref('')

watch(
  () => route.fullPath,
  async () => {
    try {
      const res = await projectApi.current()
      if (res.data.success && res.data.project) {
        projectName.value = res.data.project.name || '未命名项目'
      } else {
        projectName.value = ''
      }
    } catch {
      projectName.value = ''
    }
  },
  { immediate: true }
)
</script>

<style>
body {
  margin: 0;
  padding: 0;
}
#app {
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB',
    'Microsoft YaHei', Arial, sans-serif;
}

.el-header {
  display: flex;
  align-items: center;
  padding: 0 12px;
}

.el-menu--horizontal > .el-menu-item {
  padding: 0 14px !important;
}

.project-badge {
  margin-left: auto;
}
</style>