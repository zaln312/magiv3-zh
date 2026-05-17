<template>
  <div id="app">
    <el-container>
      <el-header>
        <el-menu
          :default-active="currentRoute"
          mode="horizontal"
          router
        >
          <el-menu-item index="/">上传图片</el-menu-item>
          <el-menu-item index="/ocr">OCR 识别</el-menu-item>
          <el-menu-item index="/predict">Predict 预测</el-menu-item>
          <el-menu-item index="/grounding">Grounding 指代</el-menu-item>
          <el-menu-item index="/prose">Prose 叙述</el-menu-item>
        </el-menu>
      </el-header>
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const currentRoute = computed(() => route.path)
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
</style>