package com.gwjjj.weiborumor.Business;


import com.alibaba.fastjson.JSONObject;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Controller
@RequestMapping("/user")
public class UserBusiness {
    @Value("${weibo.clientId}")
    private String clientId;
    @Value("${weibo.clientSecret}")
    private String clientSecret; //  应用密码
    @Value("${weibo.redirectUri}")
    private String redirectUri;  //  回调页面
    Logger logger = LoggerFactory.getLogger(Logger.class);

    @ResponseBody
    @RequestMapping("/userInfo")
    private String getUserInfo(@RequestBody JSONObject jsonParam) throws JsonProcessingException {
        String uid = (String) jsonParam.get("uid");
        String access_token = (String) jsonParam.get("access_token");
        String userInfoUrl="https://api.weibo.com/2/users/show.json?uid="+uid+"&access_token="+access_token;
        logger.info("userInfo =================  weibo api url: " + userInfoUrl);
        RestTemplate restTemplate = new RestTemplate();
        String forObject = restTemplate.getForObject(userInfoUrl, String.class);
        ObjectMapper objectMapper = new ObjectMapper();
        Map map2 = objectMapper.readValue(forObject, Map.class);
        Object profile_image_url = map2.get("profile_image_url");
        Object name = map2.get("name");
        Object followersCount = map2.get("followers_count");
        Object friendsCount = map2.get("friends_count");
        Object statusesCount = map2.get("statuses_count");
        Map<String, String> userInfoMap = new HashMap<>();
        userInfoMap.put("name",name.toString());
        userInfoMap.put("userPic",profile_image_url.toString());
        userInfoMap.put("followersCount",followersCount.toString());
        userInfoMap.put("friendsCount",friendsCount.toString());
        userInfoMap.put("statusesCount",statusesCount.toString());
        String result = new Gson().toJson(userInfoMap);
        return result;
    }

}
