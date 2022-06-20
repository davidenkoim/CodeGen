package org.springframework.boot.actuate.autoconfigure.env;

import java.util.stream.Collectors;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.actuate.autoconfigure.endpoint.condition.ConditionalOnAvailableEndpoint;
import org.springframework.boot.actuate.autoconfigure.endpoint.expose.EndpointExposure;
import org.springframework.boot.actuate.endpoint.SanitizingFunction;
import org.springframework.boot.actuate.env.EnvironmentEndpoint;
import org.springframework.boot.actuate.env.EnvironmentEndpointWebExtension;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.core.env.Environment;

/**
 * {@link EnableAutoConfiguration Auto-configuration} for the {@link EnvironmentEndpoint}.
 *
 * @author Phillip Webb
 * @author Stephane Nicoll
 * @since 2.0.0
 */
@AutoConfiguration
@ConditionalOnAvailableEndpoint(endpoint = EnvironmentEndpoint.class)
@EnableConfigurationProperties(EnvironmentEndpointProperties.class)
public class EnvironmentEndpointAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean
	public EnvironmentEndpoint environmentEndpoint(Environment environment, EnvironmentEndpointProperties properties,
			ObjectProvider<SanitizingFunction> sanitizingFunctions) {
		EnvironmentEndpoint VAR_10 = new EnvironmentEndpoint(environment,
				sanitizingFunctions.orderedStream().collect(Collectors.toList()));
		String[] keysToSanitize = properties.getKeysToSanitize();
		if (keysToSanitize != null) {
			VAR_10.setKeysToSanitize(keysToSanitize);
		}
		String[] additionalKeysToSanitize = properties.getAdditionalKeysToSanitize();
		if (additionalKeysToSanitize != null) {
			VAR_10.keysToSanitize(additionalKeysToSanitize);
		}
		return VAR_10;
	}

	@Bean
	@ConditionalOnMissingBean
	@ConditionalOnBean(EnvironmentEndpoint.class)
	@ConditionalOnAvailableEndpoint(exposure = { EndpointExposure.WEB, EndpointExposure.CLOUD_FOUNDRY })
	public EnvironmentEndpointWebExtension environmentEndpointWebExtension(EnvironmentEndpoint environmentEndpoint) {
		return new EnvironmentEndpointWebExtension(environmentEndpoint);
	}

}
