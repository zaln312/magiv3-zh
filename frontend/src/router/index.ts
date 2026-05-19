import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('../views/HomeView.vue'),
    },
    {
      path: '/settings',
      name: 'settings',
      component: () => import('../views/SettingsView.vue'),
    },
    {
      path: '/upload',
      name: 'upload',
      component: () => import('../views/UploadView.vue'),
    },
    {
      path: '/ocr',
      name: 'ocr',
      component: () => import('../views/OcrView.vue'),
    },
    {
      path: '/predict',
      name: 'predict',
      component: () => import('../views/PredictView.vue'),
    },
    {
      path: '/grounding',
      name: 'grounding',
      component: () => import('../views/GroundingView.vue'),
    },
    {
      path: '/prose',
      name: 'prose',
      component: () => import('../views/ProseView.vue'),
    },
    {
      path: '/character-reference',
      name: 'characterReference',
      component: () => import('../views/CharacterReferenceView.vue'),
    },
    {
      path: '/video-generation',
      name: 'videoGeneration',
      component: () => import('../views/VideoGenerationView.vue'),
    },
  ],
})

export default router